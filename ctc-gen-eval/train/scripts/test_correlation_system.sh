# SummEval - Consistency

python test_correlation_system.py --dataset_name summeval --aspect consistency --aligner_type bert --bert_model_type bert-base-uncased --aggr_type mean
python test_correlation_system.py --dataset_name summeval --aspect consistency --aligner_type bert --aggr_type mean
python test_correlation_system.py --dataset_name summeval --aspect consistency --aligner_type disc --disc_init ckpts/cnndm/disc.ckpt --aggr_type mean
python test_correlation_system.py --dataset_name summeval --aspect consistency --aligner_type disc --disc_init ckpts/xsum/disc.ckpt --aggr_type mean
python test_correlation_system.py --dataset_name summeval --aspect consistency --aligner_type bleurt --bleurt_init ckpts/cnndm/reg
python test_correlation_system.py --dataset_name summeval --aspect consistency --aligner_type bleurt --bleurt_init ckpts/xsum/reg

# SummEval - Relevance - 11 References

python test_correlation_system.py --dataset_name summeval --aspect relevance --aligner_type bert --bert_model_type bert-base-uncased --aggr_type mean
python test_correlation_system.py --dataset_name summeval --aspect relevance --aligner_type bert --aggr_type mean
python test_correlation_system.py --dataset_name summeval --aspect relevance --aligner_type disc --disc_init ckpts/cnndm_ref/disc.ckpt --relevance_y_x_init ckpts/cnndm/disc.ckpt --aggr_type mean
python test_correlation_system.py --dataset_name summeval --aspect relevance --aligner_type disc --disc_init ckpts/cnndm_ref/disc.ckpt --relevance_y_x_init ckpts/xsum/disc.ckpt --aggr_type mean
python test_correlation_system.py --dataset_name summeval --aspect relevance --aligner_type bleurt --bleurt_init ckpts/cnndm_ref/reg --relevance_y_x_init ckpts/cnndm/reg
python test_correlation_system.py --dataset_name summeval --aspect relevance --aligner_type bleurt --bleurt_init ckpts/cnndm_ref/reg --relevance_y_x_init ckpts/xsum/reg

# SummEval - Relevance - 1 References

python test_correlation_system.py --dataset_name summeval --aspect relevance --aligner_type bert --bert_model_type bert-base-uncased --aggr_type mean --n_references 1
python test_correlation_system.py --dataset_name summeval --aspect relevance --aligner_type bert --aggr_type mean --n_references 1
python test_correlation_system.py --dataset_name summeval --aspect relevance --aligner_type disc --disc_init ckpts/cnndm_ref/disc.ckpt --relevance_y_x_init ckpts/cnndm/disc.ckpt --aggr_type mean --n_references 1
python test_correlation_system.py --dataset_name summeval --aspect relevance --aligner_type disc --disc_init ckpts/cnndm_ref/disc.ckpt --relevance_y_x_init ckpts/xsum/disc.ckpt --aggr_type mean --n_references 1
python test_correlation_system.py --dataset_name summeval --aspect relevance --aligner_type bleurt --bleurt_init ckpts/cnndm_ref/reg --relevance_y_x_init ckpts/cnndm/reg --n_references 1
python test_correlation_system.py --dataset_name summeval --aspect relevance --aligner_type bleurt --bleurt_init ckpts/cnndm_ref/reg --relevance_y_x_init ckpts/xsum/reg --n_references 1

# Yelp

python test_correlation_system.py --dataset_name yelp --aspect preservation --aligner_type bert --bert_model_type bert-base-uncased --aggr_type mean
python test_correlation_system.py --dataset_name yelp --aspect preservation --aligner_type bert --aggr_type mean
python test_correlation_system.py --dataset_name yelp --aspect preservation --aligner_type bert --bert_model_type roberta-large-mnli --bert_num_layers 9 --aggr_type mean
python test_correlation_system.py --dataset_name yelp --aspect preservation --aligner_type disc --disc_init ckpts/yelp/disc.ckpt --aggr_type mean
python test_correlation_system.py --dataset_name yelp --aspect preservation --aligner_type bleurt --bleurt_init ckpts/yelp/reg

# PersonaChat - Engagingness

python test_correlation_system.py --dataset_name persona_chat --aspect engaging --aligner_type bert --bert_model_type bert-base-uncased --remove_stopwords --aggr_type sum --dialog_context fact_history
python test_correlation_system.py --dataset_name persona_chat --aspect engaging --aligner_type bert --remove_stopwords --aggr_type sum --dialog_context fact_history

python test_correlation_system.py --dataset_name persona_chat --aspect engaging --aligner_type disc --disc_init ckpts/persona_chat/disc_fact_history.ckpt --remove_stopwords --aggr_type sum --dialog_context fact_history
python test_correlation_system.py --dataset_name persona_chat --aspect engaging --aligner_type disc --disc_init ckpts/topical_chat/disc_fact_history.ckpt --remove_stopwords --aggr_type sum --dialog_context fact_history

python test_correlation_system.py --dataset_name persona_chat --aspect engaging --aligner_type bleurt --bleurt_init ckpts/persona_chat/reg --dialog_context fact_history
python test_correlation_system.py --dataset_name persona_chat --aspect engaging --aligner_type bleurt --bleurt_init ckpts/topical_chat/reg --dialog_context fact_history

# PersonaChat - Groundedness

python test_correlation_system.py --dataset_name persona_chat --aspect uses_knowledge --aligner_type bert --bert_model_type bert-base-uncased --remove_stopwords --aggr_type sum --dialog_context fact
python test_correlation_system.py --dataset_name persona_chat --aspect uses_knowledge --aligner_type bert --remove_stopwords --aggr_type sum --dialog_context fact

python test_correlation_system.py --dataset_name persona_chat --aspect uses_knowledge --aligner_type disc --disc_init ckpts/persona_chat_fact/disc_fact.ckpt --remove_stopwords --aggr_type sum --dialog_context fact
python test_correlation_system.py --dataset_name persona_chat --aspect uses_knowledge --aligner_type disc --disc_init ckpts/topical_chat_fact/disc_fact.ckpt --remove_stopwords --aggr_type sum --dialog_context fact

python test_correlation_system.py --dataset_name persona_chat --aspect uses_knowledge --aligner_type bleurt --bleurt_init ckpts/persona_chat_fact/reg --dialog_context fact
python test_correlation_system.py --dataset_name persona_chat --aspect uses_knowledge --aligner_type bleurt --bleurt_init ckpts/topical_chat_fact/reg --dialog_context fact

# TopicalChat - Engagingness

python test_correlation_system.py --dataset_name topical_chat --aspect engaging --aligner_type bert --bert_model_type bert-base-uncased --remove_stopwords --aggr_type sum --dialog_context fact_history
python test_correlation_system.py --dataset_name topical_chat --aspect engaging --aligner_type bert --aggr_type sum --remove_stopwords --dialog_context fact_history

python test_correlation_system.py --dataset_name topical_chat --aspect engaging --aligner_type disc --disc_init ckpts/persona_chat/disc_fact_history.ckpt --remove_stopwords --aggr_type sum --dialog_context fact_history
python test_correlation_system.py --dataset_name topical_chat --aspect engaging --aligner_type disc --disc_init ckpts/topical_chat/disc_fact_history.ckpt --remove_stopwords --aggr_type sum --dialog_context fact_history

python test_correlation_system.py --dataset_name topical_chat --aspect engaging --aligner_type bleurt --bleurt_init ckpts/persona_chat/reg --dialog_context fact_history
python test_correlation_system.py --dataset_name topical_chat --aspect engaging --aligner_type bleurt --bleurt_init ckpts/topical_chat/reg --dialog_context fact_history

# TopicalChat - Groundedness

python test_correlation_system.py --dataset_name topical_chat --aspect uses_knowledge --aligner_type bert --bert_model_type bert-base-uncased --remove_stopwords --aggr_type sum --dialog_context fact
python test_correlation_system.py --dataset_name topical_chat --aspect uses_knowledge --aligner_type bert --remove_stopwords --aggr_type sum --dialog_context fact

python test_correlation_system.py --dataset_name topical_chat --aspect uses_knowledge --aligner_type disc --disc_init ckpts/persona_chat_fact/disc_fact.ckpt --remove_stopwords --aggr_type sum --dialog_context fact
python test_correlation_system.py --dataset_name topical_chat --aspect uses_knowledge --aligner_type disc --disc_init ckpts/topical_chat_fact/disc_fact.ckpt --remove_stopwords --aggr_type sum --dialog_context fact

python test_correlation_system.py --dataset_name topical_chat --aspect uses_knowledge --aligner_type bleurt --bleurt_init ckpts/persona_chat_fact/reg --dialog_context fact
python test_correlation_system.py --dataset_name topical_chat --aspect uses_knowledge --aligner_type bleurt --bleurt_init ckpts/topical_chat_fact/reg --dialog_context fact