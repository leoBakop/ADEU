/* =========================================================
   SITE DATA — the one file to edit when the work grows.
   - LINKS:    paper / code / slides URLs used across pages
   - NEWS:     newest first; shown on the home page
   - PROJECTS: the research line (ADEU, the safety extension, ...)
               shown on /research/ and in the home-page "Next" section
   status: "published" | "in-progress" | "idea"
   ========================================================= */

window.SITE = {
  LINKS: {
    arxiv:  "https://arxiv.org/abs/2509.03219",
    pdf:    "https://arxiv.org/pdf/2509.03219",
    doi:    "https://doi.org/10.65109/HQII4250",
    acm:    "https://dl.acm.org/doi/abs/10.65109/HQII4250",
    aamasPdf: "https://ifaamas.org/Proceedings/aamas2026/pdfs/HQII4250.pdf",
    code:   "https://github.com/leoBakop/ADEU",
    slides: "assets/files/ADEU_AAMAS2026_slides.pptx"
  },

  NEWS: [
    { date: "May 2026", text: "ADEU presented at <b>AAMAS 2026</b> in Paphos, Cyprus. Slides are available on this page." },
    { date: "2026",     text: "Paper published in the AAMAS 2026 proceedings (DOI <a href=\"https://doi.org/10.65109/HQII4250\">10.65109/HQII4250</a>)." },
    { date: "Sep 2025", text: "Preprint released on <a href=\"https://arxiv.org/abs/2509.03219\">arXiv</a>; code released on <a href=\"https://github.com/leoBakop/ADEU\">GitHub</a>." }
  ],

  PROJECTS: [
    {
      id: "adeu",
      status: "published",
      title: "A Novel Framework for Uncertainty-Driven Adaptive Exploration",
      short: "ADEU",
      venue: "AAMAS 2026",
      summary: "A general framework that uses uncertainty to decide, within an episode, when an agent should explore and when it should trust what it has learned. Subsumes existing adaptive exploration methods.",
      page: "index.html",            // relative to the site root
      links: {
        arXiv: "https://arxiv.org/abs/2509.03219",
        DOI: "https://doi.org/10.65109/HQII4250",
        Code: "https://github.com/leoBakop/ADEU"
      }
    },
    {
      id: "safe-adeu",
      status: "in-progress",
      title: "Safe exploration as adaptive exploration",
      short: "Safety",
      venue: "Ongoing",
      summary: "Extending ADEU to safety-critical settings: reading safe exploration as an adaptive-exploration problem, where uncertainty about risk — not only about value — governs how far the agent strays from its policy.",
      page: null,                    // set to "projects/safe-adeu/index.html" once the page exists
      links: {}
    },
    {
      id: "higher-moments",
      status: "idea",
      title: "Guiding exploration through higher moments of the action distribution",
      short: "Higher moments",
      venue: "Future direction",
      summary: "Shaping exploration by modifying more than the spread of the action distribution.",
      page: null,
      links: {}
    },
    {
      id: "multiagent",
      status: "idea",
      title: "Multiagent adaptive exploration",
      short: "Multiagent",
      venue: "Future direction",
      summary: "Coordinating when several learning agents explore and when they exploit.",
      page: null,
      links: {}
    }
  ]
};
