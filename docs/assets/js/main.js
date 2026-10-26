/* Shared behaviour for every page. Reads window.SITE from data.js. */
(function () {
  const S = window.SITE || {};
  const root = document.body.dataset.root || "";   // "" on the home page, "../" one level down

  const STATUS = {
    "published":   { label: "Published",   cls: "" },
    "in-progress": { label: "In progress", cls: "wip" },
    "idea":        { label: "Future direction", cls: "idea" }
  };

  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  // Fill any element with data-link="arxiv|doi|code|slides|..." from SITE.LINKS
  document.querySelectorAll("[data-link]").forEach((a) => {
    const url = S.LINKS && S.LINKS[a.dataset.link];
    if (!url) return;
    a.href = /^https?:/.test(url) ? url : root + url;
  });

  // News list
  const news = document.getElementById("news-list");
  if (news && S.NEWS) {
    news.innerHTML = S.NEWS.map((n) => `<li><time>${esc(n.date)}</time><span>${n.text}</span></li>`).join("");
  }

  // Project cards (home page: exclude current project; research page: full timeline)
  function card(p, withMeta) {
    const st = STATUS[p.status] || STATUS.idea;
    const links = Object.entries(p.links || {})
      .map(([k, v]) => `<a href="${esc(v)}">${esc(k)}</a>`).join("");
    const page = p.page ? `<a class="more" href="${root + esc(p.page)}">Project page →</a>` : "";
    return `<article class="card ${st.cls}" id="${esc(p.id)}">
      <span class="tag ${st.cls}">${st.label}</span>
      ${withMeta ? `<div class="meta">${esc(p.venue)}</div>` : ""}
      <h4>${esc(p.title)}</h4>
      <p>${esc(p.summary)}</p>
      ${links ? `<div class="small-links">${links}</div>` : ""}
      ${page}
    </article>`;
  }

  const next = document.getElementById("next-cards");
  if (next && S.PROJECTS) {
    const skip = next.dataset.exclude;
    next.innerHTML = S.PROJECTS.filter((p) => p.id !== skip).map((p) => card(p, false)).join("");
  }
  const line = document.getElementById("research-line");
  if (line && S.PROJECTS) {
    line.innerHTML = S.PROJECTS.map((p) => card(p, true)).join("");
  }

  // Citation tabs + copy button
  document.querySelectorAll(".bib").forEach((box) => {
    const tabs = box.querySelectorAll(".tabs button");
    const panes = box.querySelectorAll("pre");
    tabs.forEach((t, i) => t.addEventListener("click", () => {
      tabs.forEach((x, j) => x.setAttribute("aria-selected", i === j));
      panes.forEach((p, j) => (p.hidden = i !== j));
    }));
    const btn = box.querySelector(".copy");
    if (btn) btn.addEventListener("click", async () => {
      const visible = [...panes].find((p) => !p.hidden);
      try { await navigator.clipboard.writeText(visible.innerText); btn.textContent = "Copied"; }
      catch (e) { btn.textContent = "Select & copy"; }
      setTimeout(() => (btn.textContent = "Copy"), 1600);
    });
  });

  // Highlight the nav entry of the section in view
  const navLinks = [...document.querySelectorAll('.nav a[href^="#"]')];
  if ("IntersectionObserver" in window && navLinks.length) {
    const map = new Map(navLinks.map((a) => [a.getAttribute("href").slice(1), a]));
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting && map.has(e.target.id)) {
          navLinks.forEach((a) => a.classList.remove("active"));
          map.get(e.target.id).classList.add("active");
        }
      });
    }, { rootMargin: "-40% 0px -55% 0px" });
    map.forEach((_, id) => { const el = document.getElementById(id); if (el) io.observe(el); });
  }

  const y = document.getElementById("year");
  if (y) y.textContent = new Date().getFullYear();
})();
