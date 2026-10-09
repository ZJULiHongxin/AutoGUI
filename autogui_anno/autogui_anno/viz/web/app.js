/* AutoGUI pipeline visualizer — vanilla JS, no framework, no build step.
 *
 * Consumes the Task-19 record schema (schema_version: 1). The six stage keys
 * are: input, diff, reject, annotate, verify, result. Each stage object is
 * either a dict carrying an `explainer` + content, or null when that stage did
 * not run (the pipeline exited early). The top-level `verdict` records where
 * the pipeline stopped: kept | rejected | no_change | uncheckable | inconsistent.
 *
 * Fetches only `viz_data/*.json` relative paths — never an external URL.
 */
(function () {
  "use strict";

  // Order of the six stacked stage cards and the banner flow.
  var STAGES = ["input", "diff", "reject", "annotate", "verify", "result"];
  var STAGE_TITLES = {
    input: "Input",
    diff: "Diff",
    reject: "Reject",
    annotate: "Annotate",
    verify: "Verify",
    result: "Result"
  };

  // ---- small DOM helpers -------------------------------------------------
  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function pill(text, kind) {
    return el("span", "pill pill-" + (kind || "neutral"), text);
  }

  // ---- banner flow -------------------------------------------------------
  // Each stage chip shows a per-stage badge: "ran", "stopped" (the stage the
  // pipeline halted at), or "skipped" (a later stage that never ran).
  function stageState(record, stage) {
    if (record[stage] != null) return "ran";
    // Null stage: it is the stopping point if it is the first null stage.
    var idx = STAGES.indexOf(stage);
    for (var i = 0; i < idx; i++) {
      if (record[STAGES[i]] == null) return "skipped";
    }
    return "stopped";
  }

  function renderFlow(record, flowEl) {
    clear(flowEl);
    STAGES.forEach(function (stage, i) {
      if (i > 0) flowEl.appendChild(el("span", "flow-arrow", "→"));
      var state = stageState(record, stage);
      var chip = el("span", "flow-chip flow-" + state);
      chip.appendChild(el("span", "flow-name", STAGE_TITLES[stage]));
      chip.appendChild(el("span", "flow-badge", state));
      flowEl.appendChild(chip);
    });
  }

  // ---- stage card scaffold ----------------------------------------------
  function stageCard(stage, explainer) {
    var card = el("section", "card card-" + stage);
    var head = el("div", "card-head");
    head.appendChild(el("h2", "card-title", STAGE_TITLES[stage]));
    card.appendChild(head);
    if (explainer) card.appendChild(el("p", "explainer", explainer));
    var body = el("div", "card-body");
    card.appendChild(body);
    return { card: card, body: body };
  }

  function nullCard(stage, verdict) {
    var card = el("section", "card card-null");
    card.appendChild(el("h2", "card-title", STAGE_TITLES[stage]));
    card.appendChild(el("p", "null-note",
      "This stage did not run — the pipeline stopped at “" +
      (verdict || "unknown") + "”."));
    return card;
  }

  // ---- per-stage content renderers --------------------------------------
  function renderInput(body, input, meta) {
    var hiIdx = (meta && typeof meta.target_line_idx === "number")
      ? meta.target_line_idx : -1;
    var grid = el("div", "input-grid");
    [["before", input.before], ["after", input.after]].forEach(function (pair) {
      var col = el("div", "tree-col");
      col.appendChild(el("div", "tree-col-title", pair[0]));
      var pre = el("div", "tree");
      (pair[1] || []).forEach(function (line, i) {
        var row = el("div", "tree-line" + (i === hiIdx ? " tree-hl" : ""), line);
        pre.appendChild(row);
      });
      col.appendChild(pre);
      grid.appendChild(col);
    });
    body.appendChild(grid);
  }

  function renderDiff(body, diff) {
    var counts = el("div", "diff-counts");
    counts.appendChild(pill("+" + (diff.num_added || 0) + " added", "added"));
    counts.appendChild(pill("-" + (diff.num_deleted || 0) + " deleted", "deleted"));
    body.appendChild(counts);

    var pre = el("div", "diff");
    (diff.lines || []).forEach(function (line) {
      // The builder feeds format_diff()'s output verbatim: lines are word-prefixed
      // ("Added ...", "Deleted ...", "Unchanged ...", "Repositioned ...",
      // "Before/After Attribute Update ..."), NOT +/- unified-diff markers. Classify
      // by that prefix; still accept a leading +/- as a fallback.
      var kind = "unchanged";
      if (/^(Added|Repositioned Here|After Attribute Update)\b/.test(line)) kind = "added";
      else if (/^(Deleted|Repositioned (Up|Down)ward|Before Attribute Update)\b/.test(line)) kind = "deleted";
      else {
        var marker = line.charAt(0);
        if (marker === "+") kind = "added";
        else if (marker === "-") kind = "deleted";
      }
      pre.appendChild(el("div", "diff-line diff-" + kind, line));
    });
    body.appendChild(pre);
  }

  function renderReject(body, reject) {
    var scores = reject.scores || [];
    var row = el("div", "badge-row");
    // reject.scores is a plain list of per-sample scores -> simple score/max badge.
    scores.forEach(function (s) {
      row.appendChild(pill(s + "/" + reject.max_score, "score"));
    });
    row.appendChild(reject.kept ? pill("kept", "ok") : pill("rejected", "bad"));
    body.appendChild(row);
    if (reject.reasoning) body.appendChild(el("p", "reasoning", reject.reasoning));
  }

  function renderAnnotate(body, annotate) {
    var func = el("p", "functionality",
      annotate.functionality || "(no functionality produced)");
    body.appendChild(func);
    if (annotate.mode) {
      var tagRow = el("div", "badge-row");
      tagRow.appendChild(pill("mode: " + annotate.mode, "mode"));
      body.appendChild(tagRow);
    }
    if (annotate.reasoning) body.appendChild(el("p", "reasoning", annotate.reasoning));
  }

  function renderVerify(body, verify) {
    // final_score / max_score shown prominently + consistent pill + candidate.
    var head = el("div", "verify-head");
    var fs = (verify.final_score != null) ? verify.final_score : 0;
    head.appendChild(el("span", "verify-score",
      (Math.round(fs * 100) / 100) + " / " + verify.max_score));
    head.appendChild(verify.consistent
      ? pill("consistent", "ok") : pill("inconsistent", "bad"));
    body.appendChild(head);

    if (verify.candidate) {
      var cand = el("div", "verify-candidate");
      cand.appendChild(el("span", "verify-candidate-label", "candidate: "));
      cand.appendChild(el("code", null, verify.candidate));
      body.appendChild(cand);
    }

    // verify.scores is a HISTOGRAM: index = score value, value = COUNT of
    // verifier trials that gave that score. Render it as a count-by-score
    // distribution, NEVER as if each entry were one trial's score.
    var scores = verify.scores || [];
    var maxCount = scores.reduce(function (m, c) { return Math.max(m, c); }, 0);
    if (scores.length && maxCount > 0) {
      var dist = el("div", "hist");
      dist.appendChild(el("div", "hist-caption", "verifier trials by score"));
      scores.forEach(function (count, scoreValue) {
        var rowEl = el("div", "hist-row");
        rowEl.appendChild(el("span", "hist-label", "score " + scoreValue));
        var track = el("div", "hist-track");
        var bar = el("div", "hist-bar");
        bar.style.width = (maxCount ? (count / maxCount * 100) : 0) + "%";
        track.appendChild(bar);
        rowEl.appendChild(track);
        rowEl.appendChild(el("span", "hist-count", String(count)));
        dist.appendChild(rowEl);
      });
      body.appendChild(dist);
    }
    if (verify.reasoning) body.appendChild(el("p", "reasoning", verify.reasoning));
  }

  function renderResult(body, result) {
    var grid = el("div", "result-grid");

    var predCol = el("div", "result-col");
    predCol.appendChild(el("div", "result-col-title", "Predicted functionality"));
    predCol.appendChild(el("p", "functionality",
      result.functionality || "(none)"));
    grid.appendChild(predCol);

    var gtCol = el("div", "result-col");
    gtCol.appendChild(el("div", "result-col-title", "Ground truth"));
    if (result.has_ground_truth) {
      gtCol.appendChild(el("p", "functionality", result.ground_truth || ""));
    } else {
      gtCol.appendChild(el("p", "muted",
        "No ground-truth annotation for this sample"));
    }
    grid.appendChild(gtCol);

    body.appendChild(grid);
  }

  var CONTENT_RENDERERS = {
    input: function (body, stage, record) { renderInput(body, stage, record.meta); },
    diff: function (body, stage) { renderDiff(body, stage); },
    reject: function (body, stage) { renderReject(body, stage); },
    annotate: function (body, stage) { renderAnnotate(body, stage); },
    verify: function (body, stage) { renderVerify(body, stage); },
    result: function (body, stage) { renderResult(body, stage); }
  };

  // ---- public: renderRecord(record, mountEl) ----------------------------
  function renderRecord(record, mountEl) {
    clear(mountEl);
    if (!record) return;

    var flowEl = document.getElementById("flow");
    if (flowEl) renderFlow(record, flowEl);

    var header = el("div", "record-header");
    header.appendChild(el("h2", "record-label", record.label || ""));
    var meta = el("div", "record-meta");
    meta.appendChild(pill(record.dataset || "", "dataset"));
    meta.appendChild(pill("verdict: " + (record.verdict || ""),
      verdictKind(record.verdict)));
    if (record.meta && record.meta.action_str) {
      meta.appendChild(el("span", "action-str", record.meta.action_str));
    }
    header.appendChild(meta);
    mountEl.appendChild(header);

    STAGES.forEach(function (stage) {
      var stageData = record[stage];
      if (stageData == null) {
        mountEl.appendChild(nullCard(stage, record.verdict));
        return;
      }
      var scaffold = stageCard(stage, stageData.explainer);
      CONTENT_RENDERERS[stage](scaffold.body, stageData, record);
      mountEl.appendChild(scaffold.card);
    });
  }

  function verdictKind(verdict) {
    if (verdict === "kept") return "ok";
    if (verdict === "rejected" || verdict === "inconsistent") return "bad";
    return "neutral";
  }

  // ---- public: renderIndex(index, railEl) -------------------------------
  function renderIndex(index, railEl) {
    clear(railEl);
    if (!index || !index.length) {
      railEl.appendChild(el("p", "muted", "No samples."));
      return;
    }

    // Group by dataset.
    var groups = {};
    var order = [];
    index.forEach(function (entry) {
      var ds = entry.dataset || "(unknown)";
      if (!groups[ds]) { groups[ds] = []; order.push(ds); }
      groups[ds].push(entry);
    });

    order.forEach(function (ds) {
      var group = el("div", "rail-group");
      group.appendChild(el("div", "rail-group-title", ds));
      groups[ds].forEach(function (entry) {
        var rowEl = el("button", "rail-row");
        rowEl.type = "button";
        rowEl.appendChild(el("span", "rail-label", entry.label || ""));
        rowEl.appendChild(pill(entry.verdict || "", verdictKind(entry.verdict)));
        rowEl.addEventListener("click", function () {
          selectRow(rowEl, railEl);
          loadRecord(entry.file);
        });
        rowEl.dataset.file = entry.file;
        group.appendChild(rowEl);
      });
      railEl.appendChild(group);
    });
  }

  function selectRow(rowEl, railEl) {
    var active = railEl.querySelectorAll(".rail-row.active");
    for (var i = 0; i < active.length; i++) active[i].classList.remove("active");
    rowEl.classList.add("active");
  }

  // ---- fetch + bootstrap -------------------------------------------------
  function loadRecord(file) {
    fetch("viz_data/" + file)
      .then(function (r) { return r.json(); })
      .then(function (record) {
        renderRecord(record, document.getElementById("main"));
      })
      .catch(function (err) {
        var main = document.getElementById("main");
        if (main) {
          clear(main);
          main.appendChild(el("p", "muted", "Failed to load " + file + ": " + err));
        }
      });
  }

  function boot() {
    var railEl = document.getElementById("rail");
    fetch("viz_data/index.json")
      .then(function (r) { return r.json(); })
      .then(function (index) {
        if (railEl) renderIndex(index, railEl);
        if (index && index.length) {
          loadRecord(index[0].file);
          if (railEl) {
            var first = railEl.querySelector(".rail-row");
            if (first) first.classList.add("active");
          }
        }
      })
      .catch(function (err) {
        if (railEl) {
          clear(railEl);
          railEl.appendChild(el("p", "muted", "Failed to load index: " + err));
        }
      });
  }

  // Expose the two render entry points.
  window.renderRecord = renderRecord;
  window.renderIndex = renderIndex;

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
