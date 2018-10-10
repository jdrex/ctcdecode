#include "bigram_decoder_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>

std::vector<std::pair<double, Output>> get_beam_search_result(
    const std::vector<BigramPathTrie *> &prefixes,
    size_t beam_size) {
  // allow for the post processing
  std::vector<BigramPathTrie *> space_prefixes;
  if (space_prefixes.empty()) {
    for (size_t i = 0; i < beam_size && i < prefixes.size(); ++i) {
      space_prefixes.push_back(prefixes[i]);
    }
  }

  std::sort(space_prefixes.begin(), space_prefixes.end(), bigram_prefix_compare);
  std::vector<std::pair<double, Output>> output_vecs;
  for (size_t i = 0; i < beam_size && i < space_prefixes.size(); ++i) {
    std::vector<int> output;
    std::vector<int> timesteps;
    space_prefixes[i]->get_path_vec(output, timesteps);
    Output outputs;
    outputs.tokens = output;
    outputs.timesteps = timesteps;
    std::pair<double, Output> output_pair(-space_prefixes[i]->approx_ctc,
                                               outputs);
    output_vecs.emplace_back(output_pair);
  }

  return output_vecs;
}

bool bigram_prefix_compare(const BigramPathTrie *x, const BigramPathTrie *y) {
  if (x->score == y->score) {
    if (x->character == y->character) {
      return false;
    } else {
      return (x->character < y->character);
    }
  } else {
    return x->score > y->score;
  }
}

