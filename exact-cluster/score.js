'use strict'
const cluster = require('../build/Release/cluster.node');
const Encoder = require('./encode');

function score(positives_tokens, candidates_tokens) {
  //const positives_tokens = positives.map((item) => item.tokens)
  //const candidates_tokens = candidates.map((item) => item.tokens)

  const encoder = Encoder.getEncoder(positives_tokens)
  const positives_encoded = positives_tokens.map((tokens) => Encoder.encode(tokens, encoder));
  const candidates_encoded = candidates_tokens.map((tokens) => Encoder.encode(tokens, encoder));

  const results = cluster.scoreSimilarity(candidates_encoded, positives_encoded, encoder.size);
  return results;
}

module.exports = score
