/**
 * Behavioral checks for resultClustering. Run with:
 *   npx tsx scripts/testResultClustering.ts
 */
import {
  calibrateMergeThreshold,
  clusterSearchResults,
  defaultClusteringOptions,
  expandWithCorroboratingStubs,
  mergeCorroboratingSources,
} from '../src/lib/agents/search/researcher/actions/search/resultClustering';
import { Chunk } from '../src/lib/types';

let failures = 0;

const check = (name: string, condition: boolean, detail?: string) => {
  if (condition) {
    console.log(`  ok: ${name}`);
  } else {
    failures++;
    console.error(`FAIL: ${name}${detail ? ` — ${detail}` : ''}`);
  }
};

// Deterministic pseudo-random unit vectors with controllable similarity.
const mulberry32 = (seed: number) => () => {
  seed |= 0;
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

const randomUnitVector = (dim: number, rand: () => number): number[] => {
  const v = Array.from({ length: dim }, () => rand() * 2 - 1);
  const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  return v.map((x) => x / norm);
};

/** Blend `base` with noise so cosine(base, result) ≈ targetSimilarity. */
const vectorNear = (
  base: number[],
  targetSimilarity: number,
  rand: () => number,
): number[] => {
  const noise = randomUnitVector(base.length, rand);
  // Remove the component of noise along base, then renormalize.
  const dot = noise.reduce((s, x, i) => s + x * base[i], 0);
  const orth = noise.map((x, i) => x - dot * base[i]);
  const orthNorm = Math.sqrt(orth.reduce((s, x) => s + x * x, 0));
  const orthUnit = orth.map((x) => x / orthNorm);
  const sin = Math.sqrt(1 - targetSimilarity * targetSimilarity);
  return base.map((x, i) => targetSimilarity * x + sin * orthUnit[i]);
};

const chunk = (
  id: number,
  embedding: number[],
  similarity: number,
  extra?: Record<string, any>,
): Chunk => ({
  content: `content-${id}`,
  metadata: {
    title: `Title ${id}`,
    url: `https://example.com/${id}`,
    similarity,
    embedding,
    ...extra,
  },
});

const dim = 64;

// --- Scenario A: "high-scoring" embedding model -----------------------------
// Topical results sit around 0.75 pairwise; true duplicates at ~0.97. The old
// hardcoded 0.75 cutoff would have merged nearly everything.
{
  console.log('Scenario A: topical ~0.75, duplicates ~0.97');
  const rand = mulberry32(42);
  const topic = randomUnitVector(dim, rand);

  const distinct = Array.from({ length: 8 }, (_, i) =>
    chunk(i, vectorNear(topic, 0.87, rand), 0.9 - i * 0.01),
  ); // pairwise ≈ 0.87² ≈ 0.76 between distinct members

  const dupOfFirst = [
    chunk(100, vectorNear(distinct[0].metadata.embedding, 0.98, rand), 0.85),
    chunk(101, vectorNear(distinct[0].metadata.embedding, 0.985, rand), 0.84),
  ];

  const results = [...distinct, ...dupOfFirst].sort(
    (a, b) => b.metadata.similarity - a.metadata.similarity,
  );

  const options = defaultClusteringOptions();
  const threshold = calibrateMergeThreshold(results, options);
  console.log(`  calibrated threshold: ${threshold.toFixed(3)}`);

  const clustered = clusterSearchResults(results, options);
  const representativeUrls = clustered.map((c) => c.metadata.url);

  check(
    'distinct topical results are NOT merged',
    distinct.every((d) => representativeUrls.includes(d.metadata.url)),
    `kept ${clustered.length} of ${results.length}`,
  );

  const first = clustered.find(
    (c) => c.metadata.url === distinct[0].metadata.url,
  )!;
  const corroboratingUrls = (first.metadata.corroborating ?? []).map(
    (s: { url: string }) => s.url,
  );

  check(
    'true duplicates cluster onto the representative as corroborating',
    corroboratingUrls.includes('https://example.com/100') &&
      corroboratingUrls.includes('https://example.com/101'),
    JSON.stringify(corroboratingUrls),
  );

  check(
    'input chunks are not mutated',
    results.every((r) => r.metadata.corroborating === undefined),
  );
}

// --- Scenario B: "low-scoring" embedding model -------------------------------
// Topical results sit around 0.35 pairwise; duplicates at ~0.9. A hardcoded
// 0.9 threshold barely catches them; calibration should sit well below 0.9.
{
  console.log('Scenario B: topical ~0.35, duplicates ~0.90');
  const rand = mulberry32(7);
  const topic = randomUnitVector(dim, rand);

  const distinct = Array.from({ length: 8 }, (_, i) =>
    chunk(i, vectorNear(topic, 0.6, rand), 0.9 - i * 0.01),
  ); // pairwise ≈ 0.36 between distinct members

  const dup = chunk(
    100,
    vectorNear(distinct[0].metadata.embedding, 0.9, rand),
    0.85,
  );

  const results = [...distinct, dup].sort(
    (a, b) => b.metadata.similarity - a.metadata.similarity,
  );

  const options = defaultClusteringOptions();
  const threshold = calibrateMergeThreshold(results, options);
  console.log(`  calibrated threshold: ${threshold.toFixed(3)}`);

  check(
    'threshold clamps to the configured floor for low-scoring models',
    threshold === options.minMergeSimilarity,
    `${threshold}`,
  );

  const clustered = clusterSearchResults(results, options);

  check(
    'duplicate at 0.9 is merged, distinct results are kept',
    clustered.length === distinct.length &&
      (clustered[0].metadata.corroborating?.length ?? 0) === 1,
    `kept ${clustered.length}, corroborating on top result: ${JSON.stringify(clustered[0].metadata.corroborating)}`,
  );
}

// --- Edge cases ---------------------------------------------------------------
{
  console.log('Edge cases');
  const options = defaultClusteringOptions();

  // Too few results to calibrate → fallback threshold.
  const rand = mulberry32(3);
  const few = [
    chunk(0, randomUnitVector(dim, rand), 0.9),
    chunk(1, randomUnitVector(dim, rand), 0.8),
  ];
  check(
    'falls back with too few pairs',
    calibrateMergeThreshold(few, options) === options.fallbackMergeSimilarity,
  );

  // Empty embeddings (embedding failure path) → everything kept, no crash.
  const noEmbeddings = [chunk(0, [], 1), chunk(1, [], 1), chunk(2, [], 1)];
  check(
    'results without embeddings are all kept',
    clusterSearchResults(noEmbeddings, options).length === 3,
  );

  // Corroborating cap respected.
  const base = randomUnitVector(dim, rand);
  const capResults = [
    chunk(0, base, 0.99),
    ...Array.from({ length: 7 }, (_, i) =>
      chunk(10 + i, vectorNear(base, 0.99, rand), 0.9),
    ),
  ];
  const capClustered = clusterSearchResults(capResults, options);
  check(
    'corroborating list respects maxCorroboratingPerResult',
    capClustered.length === 1 &&
      capClustered[0].metadata.corroborating.length ===
        options.maxCorroboratingPerResult,
    `${capClustered[0]?.metadata.corroborating?.length}`,
  );

  // mergeCorroboratingSources: dedup + no self-reference + cap.
  const target: Chunk = {
    content: 'a',
    metadata: {
      url: 'https://example.com/a',
      corroborating: [{ title: 'X', url: 'https://example.com/x' }],
    },
  };
  const source: Chunk = {
    content: 'a again',
    metadata: {
      url: 'https://example.com/a',
      corroborating: [
        { title: 'X', url: 'https://example.com/x' }, // duplicate
        { title: 'A', url: 'https://example.com/a' }, // self
        { title: 'Y', url: 'https://example.com/y' },
      ],
    },
  };
  const merged = mergeCorroboratingSources(target, source, 4);
  check(
    'mergeCorroboratingSources dedups and skips self-URL',
    merged.length === 2 &&
      merged.some((s) => s.url === 'https://example.com/y'),
    JSON.stringify(merged),
  );

  // expandWithCorroboratingStubs: indices, skip-existing, content-free stubs.
  const findings: Chunk[] = [
    {
      content: 'primary',
      metadata: {
        title: 'Primary',
        url: 'https://example.com/p',
        corroborating: [
          { title: 'Dup', url: 'https://example.com/d' },
          { title: 'Already own result', url: 'https://example.com/q' },
        ],
      },
    },
    {
      content: 'other',
      metadata: { title: 'Other', url: 'https://example.com/q' },
    },
  ];
  const expanded = expandWithCorroboratingStubs(findings);
  check(
    'stub inserted right after its representative with 1-based pointer',
    expanded.length === 3 &&
      expanded[1].metadata.corroborates === 1 &&
      expanded[1].metadata.url === 'https://example.com/d' &&
      expanded[1].content.includes('result 1'),
    JSON.stringify(expanded.map((e) => e.metadata.url)),
  );
  check(
    'corroborating source that is already its own result is not duplicated',
    expanded.filter((e) => e.metadata.url === 'https://example.com/q')
      .length === 1,
  );
  check(
    'representatives in expanded output no longer carry corroborating metadata',
    expanded[0].metadata.corroborating === undefined,
  );
}

if (failures > 0) {
  console.error(`\n${failures} check(s) failed`);
  process.exit(1);
}
console.log('\nAll checks passed.');
