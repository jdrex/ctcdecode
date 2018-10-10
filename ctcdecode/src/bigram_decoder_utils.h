#ifndef BIGRAM_DECODER_UTILS_H_
#define BIGRAM_DECODER_UTILS_H_

#include <utility>
#include "fst/log.h"
#include "bigram_path_trie.h"
#include "output.h"

// Get beam search result from prefixes in trie tree
std::vector<std::pair<double, Output>> get_beam_search_result(
    const std::vector<BigramPathTrie *> &prefixes,
    size_t beam_size);

// Functor for prefix comparsion
bool bigram_prefix_compare(const BigramPathTrie *x, const BigramPathTrie *y);

#endif  // DECODER_UTILS_H
