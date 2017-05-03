'use strict'

function getEncoder(trainSet){
  let tokens = {}
  let size = 0
  for (let i=0; i < trainSet.length; ++i) {
    const item = trainSet[i]
    for (let j=0; j < item.length; ++j){
      const token = item[j]
      tokens[token] = tokens[token] || (++size)
    }
  }
  return {dict:tokens, size:size+1}
}

function encode(item, encoder) {
  // Note: 0 is reserved for "undefined" token, i.e. any token not present in training
  const dict = encoder.dict
  return item.map((token) => dict[token] || 0)
}

/* TODO: Classification set should be much larger than training set, so handle tokens
present in classification but absent in training *outside* of the C++ module.*/

exports.getEncoder = getEncoder
exports.encode = encode
