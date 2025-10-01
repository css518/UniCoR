export PYTHONPATH=.
datasets=('AtCoder' 'CodeJamData' 'CSNCC' 'XLCoST' 'XCE')
modes=('nlp2code' 'code2code' 'nlp2nlp' 'nlp+code2code+code' 'remix2code')
models=('text-embedding-3-small' 'text-embedding-3-large')

CUDA_VISIBLE_DEVICES=0

output=./result-rebuttal
cache=${output}/cache
outputs=${output}/result
csv_name=${output}/model.csv
outputs_1=${output}/result-weight
csv_name_weight=${output}/model_weight.csv


for dataset in "${datasets[@]}"
do
    if [ ${dataset} == "XCE" ]; then
        langs=('C' 'C++' 'C#' 'Go' 'Java' 'Javascript' 'PHP' 'Python' 'Ruby' 'Rust' 'Scala')
    else
        langs=('java' 'python')
    fi

    for query_file in "${langs[@]}"
    do
        for candidate_file in "${langs[@]}"
        do
            for model in "${models[@]}"
            do
                for mode in "${modes[@]}"
                do
                    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python script/eval_openai.py \
                        --cache_dir ${cache} \
                        --output_dir ${outputs}\
                        --dataset ${dataset} \
                        --query_file dataset/$dataset/$query_file.jsonl \
                        --candidate_file dataset/$dataset/$candidate_file.jsonl\
                        --code_length 256 \
                        --nl_length 128 \
                        --train_batch_size 64 \
                        --eval_batch_size 64 \
                        --learning_rate 2e-5 \
                        --model ${model}\
                        --mode ${mode}\
                        --seed 123456
                done

                python script/search_best.py \
                    --query1_path ${cache}/Vector-${model}-${dataset}-${query_lang}-nlp.pkl  \
                    --query2_path ${cache}/Vector-${model}-${dataset}-${query_lang}-code.pkl \
                    --target_path ${cache}/Vector-${model}-${dataset}-${candidate_lang}-code.pkl \
                    --prefix ${model}-${dataset}-nlp+code2code-${query_lang}2${candidate_lang} \
                    --output_dir ${outputs_1}  
            done

            CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python script/eval_bm25.py \
                    --cache_dir ${cache} \
                    --output_dir ${outputs}\
                    --dataset ${dataset} \
                    --query_file ${query_file} \
                    --candidate_file ${candidate_file}\
                    --code_length 256 \
                    --nl_length 128 \
                    --train_batch_size 64 \
                    --eval_batch_size 64 \
                    --learning_rate 2e-5 \
                    --seed 123456
        done  
    done
done

python script/Analysis.py --path ${outputs} --output_dir ${csv_name}
python script/Analysis_weight.py --path ${outputs_1} --output_dir ${csv_name_weight}