# An example of running the baseline model. 
# Please modify the following variables to fit your own dataset and memory configuration. 
# ========================================================
memory_type="FullContext"
dataset_type="LOCOMO"
dataset_path="/disk/disk_4T_2/chenyijun/MIRIX/data/locomo10.json"
# dataset_path="/disk/disk_4T_2/chenyijun/memobase/docs/experiments/locomo-benchmark/dataset/locomo10_rag.json"
config_path="/disk/disk_4T_2/chenyijun/LightMem/src/lightmem/memory_toolkits/memories/configs/NaiveRAG.json"
num_workers=1
tokenizer_path="gpt-4"
log_dir="FullContext_LOCOMO_search_logs"
token_cost_prefix="token_cost"
pid_prefix="process"
ranges=(
    "0 1"
)
api_keys=(
    "sk-LU9oAwn6foP9YfDsFeC09dD8D6A24fFe8cC01aF6F301E954"
    "YOUR_API_KEY_2"
    "YOUR_API_KEY_3"
    "YOUR_API_KEY_4"
    "YOUR_API_KEY_5"
)
base_urls=(
    "https://api.gpts.vin/v1"
    "YOUR_BASE_URL_2"
    "YOUR_BASE_URL_3"
    "YOUR_BASE_URL_4"
    "YOUR_BASE_URL_5"
)

api_keys_for_image=api_keys=(
    "sk-LU9oAwn6foP9YfDsFeC09dD8D6A24fFe8cC01aF6F301E954"
    "YOUR_API_KEY_2"
    "YOUR_API_KEY_3"
    "YOUR_API_KEY_4"
    "YOUR_API_KEY_5"
)
base_urls_for_image=base_urls=(
    "https://api.gpts.vin/v1"
    "YOUR_BASE_URL_2"
    "YOUR_BASE_URL_3"
    "YOUR_BASE_URL_4"
    "YOUR_BASE_URL_5"
)

# ========================================================

[ ! -d "$log_dir" ] && mkdir -p "$log_dir"

for ((i=0; i<${#ranges[@]}; i++)); do
    read start_idx end_idx <<< "${ranges[$i]}"
    export OPENAI_API_KEY="${api_keys[$i]}" 
    export OPENAI_API_BASE="${base_urls[$i]}"
    export OPENAI_API_KEY_FOR_IMAGE="${api_keys_for_image[$i]}" 
    export OPENAI_API_BASE_FOR_IMAGE="${base_urls_for_image[$i]}"

    log_file="${log_dir}/${pid_prefix}_$((i+1))_${start_idx}_${end_idx}.log"
    token_cost_file="${token_cost_prefix}_${memory_type,,}_$((i+1))_${start_idx}_${end_idx}"
    pid_file="${log_dir}/${pid_prefix}_$((i+1)).pid"

    [ ! -f "$log_file" ] && touch "$log_file"
    # nohup python memory_construction.py \
    #     --memory-type "$memory_type" \
    #     --dataset-type "$dataset_type" \
    #     --dataset-path "$dataset_path" \
    #     --config-path "$config_path" \
    #     --num-workers "$num_workers" \
    #     --start-idx "$start_idx" \
    #     --end-idx "$end_idx" \
    #     --token-cost-save-filename "$token_cost_file" \
    #     --tokenizer-path "$tokenizer_path" \
    #     --rerun \
    #     > "$log_file" 2>&1 &

    nohup python memory_search.py \
        --memory-type "$memory_type" \
        --dataset-type "$dataset_type" \
        --dataset-path "$dataset_path" \
        --config-path "$config_path" \
        --num-workers "$num_workers" \
        --start-idx "$start_idx" \
        --end-idx "$end_idx" \
        --strict \
        --top-k 40 \
        > "$log_file" 2>&1 &
    echo $! > "$pid_file"
    sleep 10
done