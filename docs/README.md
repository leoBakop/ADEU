# ADEU — project website



```
index.html                 ADEU paper page (home)
research/index.html        research line: ADEU → safety → ...
_template/project.html     copy this to start a page for a new paper
assets/css/style.css       all styling (palette at the top)
assets/js/data.js          links, news and the list of projects  ← edit this most
assets/js/main.js          shared behaviour (rendering, copy-BibTeX, nav)
assets/img/  assets/video/ figures taken from the AAMAS slides
assets/files/              downloadable slides
.nojekyll                  tells GitHub Pages to serve files as they are
```

## Where this site lives

These files go in the `docs/` folder of the `leoBakop/ADEU` repository, with
GitHub Pages set to *Deploy from a branch → main → /docs*. The site is then served at
**https://leobakop.github.io/ADEU/**, underneath the personal homepage
(https://leobakop.github.io/, repo `leoBakop.github.io`), which links to it.

All paths are relative, so the site also works if it is moved elsewhere.

## Extending the site (e.g. the safety follow-up)

* **News item** → add an entry at the top of `NEWS` in `assets/js/data.js`.
* **New project** → add an object to `PROJECTS` in `assets/js/data.js`
  (`status`: `"published"`, `"in-progress"` or `"idea"`). It appears automatically on the
  home page ("What comes next") and on `research/`.
* **Full page for a new paper** → copy `_template/project.html` to
  `projects/<id>/index.html`, fill in the `{{ }}` placeholders, then set
  `page: "projects/<id>/index.html"` on its `PROJECTS` entry.
* **Colours / fonts** → variables at the top of `assets/css/style.css`.

