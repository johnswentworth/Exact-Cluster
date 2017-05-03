//s#include <node.h>
#include <nan.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <thread>
#include <time.h>

using namespace Nan;
using namespace v8;
using namespace std;

/* Significant TODO:
Typical elements in the subsets table contain about half the items, so typically
about half the words. With that much usage, there's not much point using a hash
table, so we'll probably get some savings by switching over to a fixed-size array.
*/

int compare (const void * a, const void * b) {
  return ( *(float*)a - *(float*)b ) > 0.0f ? 1 : -1;
}

float logsumexp(float* arr, int size) {
    float max = arr[0];
    for (int i = 0 ; i < size; ++i)
        max = arr[i] > max ? arr[i] : max;
    // Note: for best numerical stability, should sort arr here before summing.
    // I've commented that out for now, since empirically it doesn't seem to make much difference and it dominates the runtime.
    // qsort (arr, size, sizeof(float), compare);
    float total = 0.0;
    for (int i = 0 ; i < size; ++i)
        total += exp(arr[i]-max);
    return log(total) + max;
}

/* Inputs:
- function lP_s, which assigns a log probability to a subset of S
- array S of things which we want to cluster
*/
float* lP_partitioned_subset(float* lP_table, int n){
    // DP: each cell contains the sum over partitions of the corresponding subset
    float* table = new float[1 << n];
    float* aggr = new float[1 << n];
    table[0] = 0.0;
    int pivot = 1; // One-hot encoding for element under consideration
    for (int s = 1; s < (1 << n); ++s){
        if (!(pivot & s))
            pivot <<= 1; // MSB of s is pivot element
        int i = 0; int aggr_size = 0;
        do {
            // The main piece of work
            aggr[aggr_size++] = lP_table[s & ~i] + table[i];

            // Magic section (subset iteration of a subset)
            i |= ~(s ^ pivot); // Fill "spaces" with 1's so addition propagates through
            i += 1;
            i &= (s ^ pivot); // Unfill "spaces"
        } while (i);
        table[s] = logsumexp(aggr, aggr_size);
    }

    delete[] aggr;
    return table;
};

// Two lgamma LUTs: one for single token, the other for all-token normalizer.
float* precomputed_lgamma = new float[64];
void init_precomp() {
    for (int i = 0; i < 64; ++i)
        precomputed_lgamma[i] = lgamma(0.5f + i);
}

inline float lgamma_pre(int count) {
    if (count < 64)
        return precomputed_lgamma[count];
    return lgamma(0.5f + count);
}

float* precomputed_lgamma_all = new float[512];
void init_precomp_all(float numWords) {
    for (int i = 0; i < 512; ++i)
        precomputed_lgamma_all[i] = lgamma(0.5f*numWords + i);
}

inline float lgamma_all_pre(int count, float numWords) {
    if (count < 512)
        return precomputed_lgamma_all[count];
    return lgamma(0.5f*numWords + count);
}

float halfGamma = lgamma(0.5);
float lP_state_dirichlet(unordered_map<int, int>& tokenCounts, float numWords) {
    float result = 0.0;
    int totalCount = 0;
    for (auto it = tokenCounts.begin(); it != tokenCounts.end(); ++it ) {
        int count = it->second;
        result += lgamma_pre(count) -  halfGamma;
        totalCount += count;
    }
    result += lgamma_all_pre(0, numWords) - lgamma_all_pre(totalCount, numWords);
    return result;
};

inline float lP_state_dirichlet_update(unordered_map<int, int>& stateCounts, unordered_map<int, int>& updateCounts) {
    float result = 0.0;
    auto endStateCounts = stateCounts.end();
    int key; int count; int stateCount;
    for (auto it = updateCounts.begin(); it != updateCounts.end(); ++it ) {
        key = it->first;
        count = it->second;
        auto stateCountIt = stateCounts.find(key);
        if (stateCountIt != endStateCounts) {
            stateCount = stateCountIt->second;
            result += lgamma_pre(count + stateCount) - lgamma_pre(stateCount);
        } else
            result += lgamma_pre(count) -  halfGamma;
    }
    return result;
};

inline void updateState_dirichlet(unordered_map<int, int>& newCounts, vector<int>& newTokens) {
    for (auto it = newTokens.begin(); it != newTokens.end(); ++it) {
        newCounts[*it] = newCounts.count(*it) > 0 ? newCounts[*it] + 1 : 1;
    }
}

vector<int> getItemData(Local<Object> jsItem) {
    Local<Array> jsIntArray = Local<Array>::Cast(jsItem);
    int size = jsIntArray->Length();
    vector<int> result = vector<int>();
    result.reserve(size);
    for (int i = 0; i < size; ++i)
        result.push_back(jsIntArray->Get(i)->ToInteger()->Value());
    return result;
};

NAN_METHOD(scoreSimilarity) {
    auto t0 = clock();
    cout << CLOCKS_PER_SEC << endl;

    Nan::HandleScope scope;

    Local<Array> jsPositiveItems = Local<Array>::Cast(info[1]);
    int numPositiveItems = jsPositiveItems->Length();
    vector<vector<int>> positiveItems = vector<vector<int>>();
    positiveItems.reserve(numPositiveItems);
    for (int i = 0; i < numPositiveItems; ++i)
        positiveItems.push_back(getItemData(jsPositiveItems->CloneElementAt(i)));

    float numWords = Local<Number>::Cast(info[2])->Value();
    init_precomp();
    init_precomp_all(numWords);

    // 1. Build subset log probability table for positive items
    float* lP_table = new float[1 << numPositiveItems];
    unordered_map<int, int>* state_table = new unordered_map<int, int>[1 << numPositiveItems];
    float* total_table = new float[1 << numPositiveItems];
    lP_table[0] = 0.0f;
    state_table[0] = unordered_map<int, int>();
    total_table[0] = 0.0f;
    for (int i = 0; i < numPositiveItems; ++i) {
        int pivot = 1 << i; // one-hot encoding of current item
        vector<int> item = positiveItems[i];
        float numItemWords = item.size();
        for (int s = 0; s < pivot; ++s) {
            unordered_map<int, int> counts(state_table[s]);
            updateState_dirichlet(counts, item);
            state_table[pivot + s] = counts;
            lP_table[pivot + s] = lP_state_dirichlet(state_table[pivot + s], numWords);
            total_table[pivot + s] = total_table[s] + numItemWords;
        }
    }

    // 2. Build partitioned subset log probability table for positive items
    float* partition_table = lP_partitioned_subset(lP_table, numPositiveItems);

    float normalizer = partition_table[(1 << numPositiveItems) - 1];
    cout << "init time: " << clock()-t0 << endl;
    cout << "normalizer: " << normalizer << endl;

    int subsetT = 0;

    // 3. Loop through items and assign score equal to delta log odds that item is positive
    Local<Array> jsItems = Local<Array>::Cast(info[0]);
    int numItems = jsItems->Length();
    Local<Array> results = New<Array>(numItems);
    float* aggr = new float[1 << numPositiveItems];
    for (int i = 0; i < numItems; ++i) {
        vector<int> item = getItemData(jsItems->CloneElementAt(i));
        float numItemWords = item.size();
        unordered_map<int, int> itemWordCounts = unordered_map<int, int>();
        updateState_dirichlet(itemWordCounts, item);

        /*
        TODO: I think the big O(2^(# positive items)) loop can be done in O(2^(# number of positive items which share a token with this item)).
        That would require some math and probably some more precomputation to sum over the empty-intersection set components analytically.
        */

        /*
        emptyIntersections is the binary-encoded set of positive items which do not share a token with this item
        In our data, this should include most of the positive items, which enables lots of computational savings below.
        */
        int emptyIntersections = 0;
        for (int m = numPositiveItems - 1; m >= 0; --m) {
            emptyIntersections <<= 1;
            bool foundMatch = false;
            for (auto it = positiveItems[m].begin(); it < positiveItems[m].end(); ++it)
                foundMatch |= itemWordCounts.count(*it);
            emptyIntersections += !foundMatch;
        }

        auto t1 = clock();
        // Recompute lP_table, adding this item to every subset
        int ss = (1 << numPositiveItems) - 1;
        for (int s = 0; s < (1 << numPositiveItems); ++s) {
            if ((s & ~emptyIntersections) != s) {
                // If one of the positive items shares no tokens with this item, then it doesn't contribute to lP_state_dirichlet_update, so we can ignore it.
                aggr[s] = aggr[s & ~emptyIntersections];
            } else {
                aggr[s] = lP_state_dirichlet_update(state_table[s], itemWordCounts);
            }
        }
        for (int s = 0; s < (1 << numPositiveItems); ++s) {
            aggr[s] += lP_table[s] + partition_table[ss & ~s] + lgamma_all_pre(total_table[s], numWords) - lgamma_all_pre(total_table[s] + numItemWords, numWords);
        }
        float score = logsumexp(aggr, 1 << numPositiveItems) - normalizer - lP_state_dirichlet(itemWordCounts, numWords);
        subsetT += clock() - t1;

        Nan::Set(results, i, New<Number>(score));
    }

    cout << "subset time: " << subsetT << endl;

    delete[] lP_table;
    delete[] state_table;
    delete[] total_table;
    delete[] partition_table;
    delete[] aggr;

    cout << "finish time: " << clock()-t0 << endl;

    info.GetReturnValue().Set(results);
}

// node.js magic
NAN_MODULE_INIT(Init) {
    Nan::Set(target, New<String>("scoreSimilarity").ToLocalChecked(),
        GetFunction(New<FunctionTemplate>(scoreSimilarity)).ToLocalChecked());
}

NODE_MODULE(cluster, Init)
