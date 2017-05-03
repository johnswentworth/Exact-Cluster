var score = require('./score')
var testData = require('./test-data')

var test = function() {
    var start = new Date().getTime();
    results = score(testData.positives, testData.candidates)
    console.log("js-measured time:", new Date().getTime() - start);
    return results;
}
module.exports = test;

if (require.main === module) {
  console.log(test())
}
