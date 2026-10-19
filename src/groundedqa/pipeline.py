"""Question in, answer out: retrieval, the support gate, then Claude or the offline answerer."""

from .answers import MODE_CLAUDE, MODE_EXTRACTIVE, Answer, not_covered
from .claude import ClaudeAnswerer, ClaudeUnavailable, make_client
from .config import Settings
from .dense import load_embedder
from .extractive import answer_extractively
from .index import Index
from .retrieval import Retriever

NO_CREDENTIALS = (
    "No Anthropic credentials found (set ANTHROPIC_API_KEY or run `ant auth login`), "
    "so answers come from offline extractive mode."
)


class QAService:
    def __init__(self, retriever: Retriever, settings: Settings, offline: bool = False, client=None):
        self.retriever = retriever
        self.settings = settings
        self.claude: ClaudeAnswerer | None = None
        # Explains why answers are offline when the user did not ask for it.
        self.offline_note: str | None = None

        if not offline:
            client = client if client is not None else make_client()
            if client is None:
                self.offline_note = NO_CREDENTIALS
            else:
                self.claude = ClaudeAnswerer(
                    client,
                    model=settings.model,
                    contact=settings.contact,
                    effort=settings.effort,
                    max_tokens=settings.max_tokens,
                )

    @classmethod
    def from_settings(cls, settings: Settings, offline: bool = False) -> "QAService":
        index = Index.load(settings.index_dir)
        embedder = load_embedder(index.embedder_spec) if index.embedder_spec else None
        return cls(Retriever(index, settings.min_support, embedder), settings, offline=offline)

    @property
    def mode(self) -> str:
        return MODE_CLAUDE if self.claude is not None else MODE_EXTRACTIVE

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        retrieval = self.retriever.search(question, top_k or self.settings.top_k)

        if retrieval.supported and self.claude is not None:
            try:
                answer = self.claude.answer(question, retrieval.hits)
            except ClaudeUnavailable as exc:
                # Stay offline for the rest of the session instead of failing on every question.
                self.claude = None
                self.offline_note = f"{exc} Switched to offline extractive mode."
            else:
                answer.retrieval = retrieval
                return answer

        if not retrieval.supported:
            answer = not_covered(question, self.mode, self.settings.contact, retrieval)
        else:
            answer = answer_extractively(retrieval, self.retriever.idf, self.settings.contact)
        if self.claude is None and self.offline_note:
            answer.note = self.offline_note
        return answer
