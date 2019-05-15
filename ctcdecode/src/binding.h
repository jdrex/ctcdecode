
int paddle_beam_decode(THFloatTensor *th_probs,
                       THIntTensor *th_seq_lens,
                       const char* labels[],
		       int *label_lens,
                       int vocab_size,
                       size_t beam_size,
                       size_t num_processes,
                       double cutoff_prob,
                       size_t cutoff_top_n,
                       size_t blank_id,
		       int max_overlap, 
		       int bigram,
		       int subword,
                       THIntTensor *th_output,
                       THIntTensor *th_timesteps,
                       THFloatTensor *th_scores,
                       THIntTensor *th_out_length);

int paddle_beam_decode_lm(THFloatTensor *th_probs,
                          THIntTensor *th_seq_lens,
                          const char* labels[],
			  int *label_lens,
                          int vocab_size,
                          size_t beam_size,
                          size_t num_processes,
                          double cutoff_prob,
                          size_t cutoff_top_n,
                          size_t blank_id,
			  int max_overlap,
			  int bigram,
			  int subword,
                          void *scorer,
                          THIntTensor *th_output,
                          THIntTensor *th_timesteps,
                          THFloatTensor *th_scores,
                          THIntTensor *th_out_length);

void* paddle_get_scorer(double alpha,
                        double beta,
                        const char* lm_path,
                        const char* labels[],
			int *label_lens,
                        int vocab_size);

int is_character_based(void *scorer);
size_t get_max_order(void *scorer);
size_t get_dict_size(void *scorer);
void reset_params(void *scorer, double alpha, double beta);
