conda activate nlp

$GIT_REP = "C:\Users\loicf\Documents\IRISA\BertPlausibilityStudy"
cd $GIT_REP

python .\inference_scripts\e_snli\entropy_study_mean_head_agreg.py --batch_size 4
python .\inference_scripts\e_snli\entropy_study_sum_head_agreg.py --batch_size 4
python .\inference_scripts\e_snli\entropy_study_mean_evw_agreg.py --batch_size 4