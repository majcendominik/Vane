import computeSimilarity from '@/lib/utils/computeSimilarity';
import { Chunk } from '@/lib/types';

/**
 * Near-duplicate handling for search results.
 *
 * Instead of discarding near-duplicate results (upstream behaviour), they are
 * clustered: one representative keeps its content for the LLM to read, and the
 * merged-away results survive as citation-only references on that
 * representative ("corroborating sources"). The writer can then cite the whole
 * group, e.g. [1][2][3], while only one copy of the content spends tokens.
 *
 * The merge threshold is calibrated per batch rather than hardcoded, because
 * absolute cosine thresholds are not comparable across embedding models: true
 * duplicates (syndicated / mirrored snippets) show up as outliers above the
 * batch's own topical-similarity distribution, wherever that distribution
 * happens to sit for the configured model.
 *
 * All functions are pure with respect to their inputs (input chunks are never
 * mutated) and all tuning is passed explicitly via {@link ClusteringOptions}.
 */

export type CorroboratingSource = {
  title: string;
  url: string;
};

export type ClusteringOptions = {
  /**
   * How many standard deviations above the batch's mean pairwise similarity a
   * pair must sit to be considered the same underlying content.
   */
  outlierDeviations: number;
  /**
   * Clamp for the calibrated threshold. The floor stops over-merging when a
   * batch is uniformly dissimilar (tiny std); the ceiling guarantees that
   * verbatim copies are always merged.
   */
  minMergeSimilarity: number;
  maxMergeSimilarity: number;
  /**
   * Threshold used when the batch has too few comparable pairs to estimate a
   * distribution from.
   */
  fallbackMergeSimilarity: number;
  /** Minimum number of comparable pairs required to calibrate. */
  minPairsForCalibration: number;
  /** Cap on citation-only references attached to a single representative. */
  maxCorroboratingPerResult: number;
};

/** Fresh defaults per call site — deliberately a function, not a shared object. */
export const defaultClusteringOptions = (): ClusteringOptions => ({
  outlierDeviations: 2.5,
  minMergeSimilarity: 0.8,
  maxMergeSimilarity: 0.95,
  fallbackMergeSimilarity: 0.9,
  minPairsForCalibration: 10,
  maxCorroboratingPerResult: 4,
});

const embeddingOf = (chunk: Chunk): number[] | null => {
  const embedding = chunk.metadata?.embedding;
  return Array.isArray(embedding) && embedding.length > 0 ? embedding : null;
};

/**
 * Calibrates the duplicate-merge threshold from the batch's own pairwise
 * similarity distribution: mean + `outlierDeviations` * std, clamped to
 * [minMergeSimilarity, maxMergeSimilarity]. Falls back to
 * `fallbackMergeSimilarity` when fewer than `minPairsForCalibration`
 * comparable pairs exist.
 */
export const calibrateMergeThreshold = (
  chunks: Chunk[],
  options: ClusteringOptions,
): number => {
  const embeddings = chunks
    .map(embeddingOf)
    .filter((e): e is number[] => e !== null);

  const pairSimilarities: number[] = [];

  for (let i = 0; i < embeddings.length; i++) {
    for (let j = i + 1; j < embeddings.length; j++) {
      pairSimilarities.push(computeSimilarity(embeddings[i], embeddings[j]));
    }
  }

  if (pairSimilarities.length < options.minPairsForCalibration) {
    return options.fallbackMergeSimilarity;
  }

  const mean =
    pairSimilarities.reduce((sum, s) => sum + s, 0) / pairSimilarities.length;
  const variance =
    pairSimilarities.reduce((sum, s) => sum + (s - mean) ** 2, 0) /
    pairSimilarities.length;
  const threshold = mean + options.outlierDeviations * Math.sqrt(variance);

  return Math.min(
    options.maxMergeSimilarity,
    Math.max(options.minMergeSimilarity, threshold),
  );
};

const cloneChunk = (chunk: Chunk): Chunk => ({
  content: chunk.content,
  metadata: { ...chunk.metadata },
});

const addCorroboratingSource = (
  representative: Chunk,
  duplicate: Chunk,
  maxPerResult: number,
): void => {
  const url = duplicate.metadata?.url;
  // Same-URL entries carry no extra citation value (the researcher's URL
  // merge handles those); untitled/urlless entries cannot be cited.
  if (!url || url === representative.metadata.url) return;

  const existing: CorroboratingSource[] =
    representative.metadata.corroborating ?? [];

  if (existing.length >= maxPerResult) return;
  if (existing.some((source) => source.url === url)) return;

  representative.metadata.corroborating = [
    ...existing,
    { title: String(duplicate.metadata?.title ?? ''), url },
  ];
};

/**
 * Greedy clustering over results sorted by descending relevance (callers sort
 * beforehand, so the most relevant member of each cluster becomes its
 * representative). Returns cloned representatives, each carrying its merged
 * neighbours under `metadata.corroborating`. Results without embeddings are
 * kept as their own representatives, matching upstream behaviour.
 */
export const clusterSearchResults = (
  results: Chunk[],
  options: ClusteringOptions,
): Chunk[] => {
  const mergeThreshold = calibrateMergeThreshold(results, options);
  const representatives: Chunk[] = [];

  for (const result of results) {
    const embedding = embeddingOf(result);

    const matchedRepresentative = embedding
      ? representatives.find((representative) => {
          const repEmbedding = embeddingOf(representative);
          return (
            repEmbedding !== null &&
            computeSimilarity(embedding, repEmbedding) > mergeThreshold
          );
        })
      : undefined;

    if (matchedRepresentative) {
      addCorroboratingSource(
        matchedRepresentative,
        result,
        options.maxCorroboratingPerResult,
      );
    } else {
      representatives.push(cloneChunk(result));
    }
  }

  return representatives;
};

/**
 * Merges the corroborating lists of two chunks that turned out to reference
 * the same URL (used by the researcher's cross-iteration URL merge). Returns
 * the merged list; does not mutate either input.
 */
export const mergeCorroboratingSources = (
  target: Chunk,
  source: Chunk,
  maxPerResult: number,
): CorroboratingSource[] => {
  const merged: CorroboratingSource[] = [
    ...(target.metadata?.corroborating ?? []),
  ];

  for (const candidate of source.metadata?.corroborating ?? []) {
    if (merged.length >= maxPerResult) break;
    if (candidate.url === target.metadata?.url) continue;
    if (merged.some((existing) => existing.url === candidate.url)) continue;
    merged.push(candidate);
  }

  return merged;
};

/**
 * Flattens clustered findings into the writer/UI source list: each
 * representative is followed by one stub chunk per corroborating source. Stubs
 * get real citation indices (so the writer can cite the group and the UI can
 * render every source card) but only a pointer as content, so they cost the
 * writer a single line each. `metadata.corroborates` holds the 1-based index
 * of the representative in the returned array.
 *
 * Corroborating sources whose URL already appears as a result of its own are
 * skipped to avoid duplicate source cards.
 */
export const expandWithCorroboratingStubs = (findings: Chunk[]): Chunk[] => {
  const representativeUrls = new Set(
    findings.map((f) => f.metadata?.url).filter(Boolean),
  );

  const expanded: Chunk[] = [];

  for (const finding of findings) {
    const representative = cloneChunk(finding);
    const corroborating: CorroboratingSource[] =
      representative.metadata.corroborating ?? [];
    delete representative.metadata.corroborating;

    expanded.push(representative);
    const representativeIndex = expanded.length; // 1-based

    for (const source of corroborating) {
      if (representativeUrls.has(source.url)) continue;
      representativeUrls.add(source.url);

      expanded.push({
        content: `Reports the same finding as result ${representativeIndex}; contains no additional content. Cite together with result ${representativeIndex}.`,
        metadata: {
          title: source.title,
          url: source.url,
          corroborates: representativeIndex,
        },
      });
    }
  }

  return expanded;
};
