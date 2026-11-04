// Extracts the comments currently loaded in the DOM.
function extractComments(limit = 500, includeReplies = true) {
  const nodes = document.querySelectorAll("ytd-comment-thread-renderer #content-text");
  const seen = new Set();
  const out = [];
  for (const node of nodes) {
    if (!includeReplies && node.closest("ytd-comment-replies-renderer")) continue;
    const text = node.textContent.trim();
    if (text && !seen.has(text)) {
      seen.add(text);
      out.push(text);
      if (out.length >= limit) break;
    }
  }
  return out;
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "EXTRACT_COMMENTS") {
    sendResponse({
      comments: extractComments(msg.limit || 500, msg.includeReplies !== false),
    });
  } else if (msg.type === "SCROLL_FOR_COMMENTS") {
    window.scrollBy(0, window.innerHeight * 3);
    sendResponse({ ok: true });
  }
  return true;
});