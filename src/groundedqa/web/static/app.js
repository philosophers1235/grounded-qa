"use strict";

const form = document.getElementById("ask-form");
const input = document.getElementById("question");
const button = document.getElementById("ask-button");
const answerSection = document.querySelector(".answer");
const answerBody = document.getElementById("answer-body");
const answerHeading = document.getElementById("answer-heading");
const sourceList = document.getElementById("source-list");
const sourcesEmpty = document.getElementById("sources-empty");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

async function ask(question) {
  button.disabled = true;
  button.textContent = "Asking...";
  const url = new URL(window.location);
  url.searchParams.set("q", question);
  history.replaceState(null, "", url);

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await response.json();
    if (!response.ok) {
      const detail = typeof data.detail === "string" ? data.detail : "Type a question first.";
      throw new Error(detail);
    }
    renderAnswer(data);
  } catch (error) {
    renderError(error.message || "The request failed. Check that the server is running.");
  } finally {
    button.disabled = false;
    button.textContent = "Ask";
  }
}

function renderAnswer(data) {
  answerSection.classList.toggle("refused", data.refused);
  answerHeading.textContent = data.refused ? "Not in the documents" : "Answer";
  answerBody.replaceChildren();

  const text = el("p", "answer-text");
  for (const part of data.parts) {
    if (!part.sources.length) {
      text.append(part.text);
      continue;
    }
    const body = part.text.trimEnd();
    text.append(body, " ");
    for (const number of part.sources) {
      const marker = el("button", "marker", String(number));
      marker.type = "button";
      marker.dataset.source = number;
      marker.setAttribute("aria-label", `Show source ${number}`);
      text.append(marker);
    }
    text.append(part.text.slice(body.length));
  }
  answerBody.append(text);

  if (data.mode === "claude" && !data.refused) {
    answerBody.append(el("p", "answer-meta", `Written by Claude (${data.model}) from the cited passages.`));
  }
  if (data.note) answerBody.append(el("p", "note", data.note));

  renderSources(data.sources);
}

function renderSources(sources) {
  sourceList.replaceChildren();
  sourcesEmpty.hidden = sources.length > 0;
  sourcesEmpty.textContent = "No passage in the documents answers this question.";

  for (const source of sources) {
    const item = el("li", "source");
    item.id = `source-${source.number}`;
    item.tabIndex = -1;

    item.append(el("span", "source-number", String(source.number)));
    const body = el("div", "source-body");
    body.append(el("div", "source-title", source.doc_title));
    const where = [source.heading, source.page ? `page ${source.page}` : ""].filter(Boolean).join(", ");
    if (where) body.append(el("div", "source-where", where));

    const passage = el("blockquote", "passage");
    let cursor = 0;
    for (const [start, end] of mergeSpans(source.highlights)) {
      passage.append(source.text.slice(cursor, start));
      passage.append(el("mark", "", source.text.slice(start, end)));
      cursor = end;
    }
    passage.append(source.text.slice(cursor));
    body.append(passage);
    body.append(el("div", "source-path", source.doc_path));
    item.append(body);
    sourceList.append(item);
  }
}

function mergeSpans(spans) {
  const merged = [];
  for (const [start, end] of [...spans].sort((a, b) => a[0] - b[0])) {
    const last = merged[merged.length - 1];
    if (last && start <= last[1]) last[1] = Math.max(last[1], end);
    else merged.push([start, end]);
  }
  return merged;
}

function renderError(message) {
  answerSection.classList.remove("refused");
  answerHeading.textContent = "Answer";
  answerBody.replaceChildren(el("p", "error", message));
  sourceList.replaceChildren();
  sourcesEmpty.hidden = true;
}

function showSource(number) {
  const target = document.getElementById(`source-${number}`);
  if (!target) return;
  for (const node of document.querySelectorAll(".source.active, .marker.active")) {
    node.classList.remove("active");
  }
  target.classList.add("active");
  for (const marker of document.querySelectorAll(`.marker[data-source="${number}"]`)) {
    marker.classList.add("active");
  }
  // Long passages scroll inside the list, so bring the highlighted words into view, not just the card.
  const mark = target.querySelector("mark") || target;
  mark.scrollIntoView({ block: "center", behavior: reducedMotion ? "auto" : "smooth" });
  target.focus({ preventScroll: true });
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (question) ask(question);
});

answerBody.addEventListener("click", (event) => {
  const marker = event.target.closest(".marker");
  if (marker) showSource(marker.dataset.source);
});

for (const example of document.querySelectorAll(".example")) {
  example.addEventListener("click", () => {
    input.value = example.dataset.question;
    ask(input.value);
  });
}

const initial = new URLSearchParams(window.location.search).get("q");
if (initial) {
  input.value = initial;
  ask(initial);
}
