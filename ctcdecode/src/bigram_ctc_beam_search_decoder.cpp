#include "bigram_ctc_beam_search_decoder.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <utility>

#include "bigram_decoder_utils.h"
#include "decoder_utils.h"
#include "ThreadPool.h"
#include "fst/fstlib.h"
#include "bigram_path_trie.h"

using FSTMATCH = fst::SortedMatcher<fst::StdVectorFst>;

std::vector<std::pair<double, Output>> bigram_ctc_beam_search_decoder(
    const std::vector<std::vector<double>> &probs_seq,
    const std::vector<std::string> &vocabulary,
    size_t beam_size,
    double cutoff_prob,
    size_t cutoff_top_n,
    size_t blank_id,
    Scorer *ext_scorer) {
  // dimension check
  size_t num_time_steps = probs_seq.size();
  for (size_t i = 0; i < num_time_steps; ++i) {
    // JD: this won't be true - vocabulary = blank, space, [chars] - seq = vocab.size() + (vocab.size()-2)^2
    VALID_CHECK_EQ(probs_seq[i].size(),
                   vocabulary.size() + (vocabulary.size()-2)*(vocabulary.size()-2),
                   "The shape of probs_seq does not match with "
                   "the shape of the vocabulary");
  }

  // was commented out before...
  // assign blank id
  // size_t blank_id = vocabulary.size();

  // JD: this is what it should be:
  //    blank = 0
  //    space = 1
  //    [chars] = 2 to (nChars+2)
  //    bigrams: (i_j) = (nChars+2) + i*nChars + j
  //size_t blank_id = 0;
  size_t space_id = 1;
  size_t nChars = vocabulary.size() - 2;

  // original:
  // assign space id
  //auto it = std::find(vocabulary.begin(), vocabulary.end(), " ");
  //int space_id = it - vocabulary.begin();
  // if no space in vocabulary
  //if ((size_t)space_id >= vocabulary.size()) {
  //  space_id = -2;
  //}

  // init prefixes' root
  BigramPathTrie root;
  root.score = root.log_prob_b_prev = 0.0;
  std::vector<BigramPathTrie *> prefixes;
  prefixes.push_back(&root);

  //if (ext_scorer != nullptr && !ext_scorer->is_character_based()) {
  //  auto fst_dict = static_cast<fst::StdVectorFst *>(ext_scorer->dictionary);
  //  fst::StdVectorFst *dict_ptr = fst_dict->Copy(true);
  //  root.set_dictionary(dict_ptr);
  //  auto matcher = std::make_shared<FSTMATCH>(*dict_ptr, fst::MATCH_INPUT);
  //  root.set_matcher(matcher);
  //}

  // prefix search over time
  for (size_t time_step = 0; time_step < num_time_steps; ++time_step) {
    auto &prob = probs_seq[time_step];

    float min_cutoff = -NUM_FLT_INF;
    bool full_beam = false;
    //if (ext_scorer != nullptr) {
    //  size_t num_prefixes = std::min(prefixes.size(), beam_size);
    //  std::sort(
    //      prefixes.begin(), prefixes.begin() + num_prefixes, prefix_compare);
    //  min_cutoff = prefixes[num_prefixes - 1]->score +
    //               std::log(prob[blank_id]) - std::max(0.0, ext_scorer->beta);
    //  full_beam = (num_prefixes == beam_size);
    //}

    std::vector<std::pair<size_t, float>> log_prob_idx =
        get_pruned_log_probs(prob, cutoff_prob, cutoff_top_n);
    // loop over chars
    for (size_t index = 0; index < log_prob_idx.size(); index++) {
      auto c = log_prob_idx[index].first;
      auto log_prob_c = log_prob_idx[index].second;
      int c_1 = -1;
      int c_2 = -1;
	
      // JD: figure out if this is a single character or a bigram (if index is > len(vocab) + 2
      bool is_bigram = (c >= vocabulary.size());
      if (is_bigram) {
	int c_1 = (c - vocabulary.size()) / nChars;
	int c_2 = (c - vocabulary.size()) % nChars;  
      }

      for (size_t i = 0; i < prefixes.size() && i < beam_size; ++i) {
        auto prefix = prefixes[i];
        if (full_beam && log_prob_c + prefix->score < min_cutoff) {
          break;
        }
        // blank
        if (c == blank_id) {
          prefix->log_prob_b_cur =
              log_sum_exp(prefix->log_prob_b_cur, log_prob_c + prefix->score);
          continue;
        }
        // JD: repeated single character
        if (c == prefix->character) {
          prefix->log_prob_nb_cur = log_sum_exp(prefix->log_prob_nb_cur, log_prob_c + prefix->log_prob_nb_prev);
        }

	// JD: repeated bigram
	if (is_bigram && (c_2 == prefix->character && c_1 == prefix->parent->character)){
          prefix->log_prob_nb_2_cur = log_sum_exp(prefix->log_prob_nb_2_cur, log_prob_c + prefix->log_prob_nb_2_prev);
	}

        // get new prefix
	// JD: first character in bigram = last character in prefix vs. new single char vs. new bigram
        // *** in bigram case there could be two new prefixes
	// e.g. prefix ends in AB, bigram is BC --> ABC or ABBC
	// single character case, AB followed by B can only by ABB
	// bigram case, AB followed by CD can only be ABCD

	if (!is_bigram) {
	  auto prefix_new = prefix->get_path_trie(c, time_step);
	  if (prefix_new != nullptr) {
	    float log_p = -NUM_FLT_INF;

	    if (c == prefix->character &&
		prefix->log_prob_b_prev > -NUM_FLT_INF) {
	      // JD: single character same as previous, so if we're repeating it the previous has to end in blank
	      log_p = log_prob_c + prefix->log_prob_b_prev;
	    } else if (c != prefix->character) {
	      // JD: single character is new
	      log_p = log_prob_c + prefix->score;
	    }
	    
	    prefix_new->log_prob_nb_cur =
	      log_sum_exp(prefix_new->log_prob_nb_cur, log_p);
	  }
	  
	} else {
	  // JD need get_path_trie function to take two characters/two steps at once
	  // prefix_new will be what happens when you add the full new character to the prefix
	  // e.g. AB + BC = ABBC
	  auto prefix_tmp = prefix->get_path_trie(c_1, time_step);
	  auto prefix_new = prefix_tmp->get_path_trie(c_2, time_step);

	  if (prefix_new != nullptr) {
	    float log_p = -NUM_FLT_INF;

	    if (c_2 == prefix->character && c_1 == prefix->parent->character){
	      // JD: repeated bigram (AB AB -> ABAB)
	      log_p = log_prob_c + prefix->log_prob_b_prev;
	    } else {
	      // JD: new bigram (A AB --> AAB or AB CD --> ABCD)
	      log_p = log_prob_c + prefix->score;
	    }
	    
	    prefix_new->log_prob_nb_2_cur =
              log_sum_exp(prefix_new->log_prob_nb_2_cur, log_p);
	  }
	  
	  if (c_1 == prefix->character){
	    //this is AB + BC = ABC
	    auto prefix_new_2 = prefix->get_path_trie(c_2, time_step);

	    if (prefix_new_2 != nullptr) {
	      float log_p_2 = -NUM_FLT_INF;
	      
	      // JD: repeated last char of bigram (A AB --> AB)
	      // previous CAN'T end in blank
	      log_p_2 = log_prob_c + prefix->log_prob_nb_prev + prefix->log_prob_nb_2_prev;

	      prefix_new_2->log_prob_nb_2_cur =
		log_sum_exp(prefix_new_2->log_prob_nb_2_cur, log_p_2);

	    }
	  }

          // language model scoring - JD: ignoring this for now
          //if (ext_scorer != nullptr &&
          //    (c == space_id || ext_scorer->is_character_based())) {
          //  BigramPathTrie *prefix_to_score = nullptr;
          //  // skip scoring the space
          //  if (ext_scorer->is_character_based()) {
          //    prefix_to_score = prefix_new;
          //  } else {
          //    prefix_to_score = prefix;
          //  }

          //  float score = 0.0;
          //  std::vector<std::string> ngram;
          //  ngram = ext_scorer->make_ngram(prefix_to_score);
          //  score = ext_scorer->get_log_cond_prob(ngram) * ext_scorer->alpha;
          //  log_p += score;
          //  log_p += ext_scorer->beta;
          //}
	    
	}
      }  // end of loop over prefix
    }    // end of loop over vocabulary


    prefixes.clear();
    // update log probs
    root.iterate_to_vec(prefixes);

    // only preserve top beam_size prefixes
    if (prefixes.size() >= beam_size) {
      std::nth_element(prefixes.begin(),
                       prefixes.begin() + beam_size,
                       prefixes.end(),
                       bigram_prefix_compare);
      for (size_t i = beam_size; i < prefixes.size(); ++i) {
        prefixes[i]->remove();
      }
    }
  }  // end of loop over time

  // score the last word of each prefix that doesn't end with space
  // JD - ignore this some more (or does it just work??)
  //if (ext_scorer != nullptr && !ext_scorer->is_character_based()) {
  //  for (size_t i = 0; i < beam_size && i < prefixes.size(); ++i) {
  //    auto prefix = prefixes[i];
  //    if (!prefix->is_empty() && prefix->character != space_id) {
  //      float score = 0.0;
  //      std::vector<std::string> ngram = ext_scorer->make_ngram(prefix);
  //      score = ext_scorer->get_log_cond_prob(ngram) * ext_scorer->alpha;
  //      score += ext_scorer->beta;
  //      prefix->score += score;
  //    }
  //  }
  //}

  size_t num_prefixes = std::min(prefixes.size(), beam_size);
  std::sort(prefixes.begin(), prefixes.begin() + num_prefixes, bigram_prefix_compare);

  // compute aproximate ctc score as the return score, without affecting the
  // return order of decoding result. To delete when decoder gets stable.
  for (size_t i = 0; i < beam_size && i < prefixes.size(); ++i) {
    double approx_ctc = prefixes[i]->score;
    //if (ext_scorer != nullptr) {
    //  std::vector<int> output;
    //  std::vector<int> timesteps;
    //  prefixes[i]->get_path_vec(output, timesteps);
    //  auto prefix_length = output.size();
    //  auto words = ext_scorer->split_labels(output);
    //  // remove word insert
    //  approx_ctc = approx_ctc - prefix_length * ext_scorer->beta;
    //  // remove language model weight:
    //  approx_ctc -= (ext_scorer->get_sent_log_prob(words)) * ext_scorer->alpha;
    //}
    prefixes[i]->approx_ctc = approx_ctc;
  }

  return get_beam_search_result(prefixes, beam_size);
}


std::vector<std::vector<std::pair<double, Output>>>
bigram_ctc_beam_search_decoder_batch(
    const std::vector<std::vector<std::vector<double>>> &probs_split,
    const std::vector<std::string> &vocabulary,
    size_t beam_size,
    size_t num_processes,
    double cutoff_prob,
    size_t cutoff_top_n,
    size_t blank_id,
    Scorer *ext_scorer) {
  VALID_CHECK_GT(num_processes, 0, "num_processes must be nonnegative!");
  // thread pool
  ThreadPool pool(num_processes);
  // number of samples
  size_t batch_size = probs_split.size();

  // enqueue the tasks of decoding
  std::vector<std::future<std::vector<std::pair<double, Output>>>> res;
  for (size_t i = 0; i < batch_size; ++i) {
    res.emplace_back(pool.enqueue(bigram_ctc_beam_search_decoder,
                                  probs_split[i],
                                  vocabulary,
                                  beam_size,
                                  cutoff_prob,
                                  cutoff_top_n,
                                  blank_id,
                                  ext_scorer));
  }

  // get decoding results
  std::vector<std::vector<std::pair<double, Output>>> batch_results;
  for (size_t i = 0; i < batch_size; ++i) {
    batch_results.emplace_back(res[i].get());
  }
  return batch_results;
}
