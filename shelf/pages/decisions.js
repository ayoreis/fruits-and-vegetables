// The decision rule from shelf/decisions.py, for the pages: turns a run's saved probabilities into answers with the
// current biases, so a changed bias shows straight away on every photo of its area. Keep the two in step.

const DECISION_LEVELS = ["empty", "half", "full"];
const DECISION_VALUES = { empty: 0, half: 0.5, full: 1 };

// Reweight the stock probabilities: lower leans toward lower levels (negative: higher), half widens "half" from
// both sides (negative: narrower).
function adjustedStock(stock, lower = 0, half = 0) {
  const weights = {};
  let total = 0;
  DECISION_LEVELS.forEach((level, i) => {
    weights[level] = stock[level] * Math.exp(lower * (DECISION_LEVELS.length - 1 - i) + (level === "half" ? half : 0));
    total += weights[level];
  });
  for (const level of DECISION_LEVELS) weights[level] /= total;
  return weights;
}

// Expected points of an answer under the challenge's scoring; an empty area always counts as compliant.
function expectedPoints(answer, compliant, stock) {
  const empty = stock.empty ?? 0;
  const otherwise = empty >= 1 ? compliant : Math.min(1, Math.max(0, (compliant - empty) / (1 - empty)));
  let total = 0;
  for (const level of DECISION_LEVELS) {
    const compliantChance = level === "empty" ? 1 : otherwise;
    for (const [actual, chance] of [[true, compliantChance], [false, 1 - compliantChance]]) {
      const identification = answer.planogram_compliant === actual ? 1 : -2;
      const value = DECISION_VALUES[level];
      const stockPoints = answer.stock_value === value ? 1 : answer.stock_value < value ? -2 : -3;
      total += chance * stock[level] * (identification + stockPoints);
    }
  }
  return total;
}

// The answer with the most expected points; candidates in the same order as Python, so ties break the same way.
function decideAnswer(compliant, stock, lower = 0, half = 0) {
  if (lower || half) stock = adjustedStock(stock, lower, half);
  let best = null;
  let bestPoints = -Infinity;
  for (const isCompliant of [true, false]) {
    for (const level of DECISION_LEVELS) {
      if (!isCompliant && level === "empty") continue;
      const answer = { planogram_compliant: isCompliant, stock_level: level, stock_value: DECISION_VALUES[level] };
      const points = expectedPoints(answer, compliant, stock);
      if (points > bestPoints) {
        best = answer;
        bestPoints = points;
      }
    }
  }
  return best;
}

// An area's own bias as {lower, half}; the file also allows a plain number, meaning lower only.
function areaBias(biases, camera, areaId) {
  const bias = biases?.[camera]?.[areaId];
  if (typeof bias === "number") return { lower: bias, half: 0 };
  return { lower: bias?.lower ?? 0, half: bias?.half ?? 0 };
}

// Every area's answer from a run's probabilities with its biases (the run's own plus each area's); areas the run
// didn't save probabilities for keep their saved answer.
function decideAll(probabilities, biases, saved = {}) {
  const result = structuredClone(saved);
  for (const [camera, files] of Object.entries(probabilities.areas)) {
    for (const [file, areas] of Object.entries(files)) {
      for (const [areaId, p] of Object.entries(areas)) {
        const own = areaBias(biases, camera, areaId);
        ((result[camera] ??= {})[file] ??= {})[areaId] = decideAnswer(
          p.compliant, p.stock, (probabilities.lower_bias ?? 0) + own.lower, (probabilities.half_bias ?? 0) + own.half,
        );
      }
    }
  }
  return result;
}
