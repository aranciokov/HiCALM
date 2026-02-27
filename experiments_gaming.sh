# experiments for figure 9c
for w in 0. 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 ; do python run_main_HierArtEx_v2_add_gaming.py --generalist clip  --room_txt_agg rnn --room_vis_agg rnn --bs 8 --lr 7e-5  --room-loss-weight $w ; done
for w in 0. 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 ; do python eval_main_HierArtEx_v2_add_gaming.py --generalist clip  --room_txt_agg rnn --room_vis_agg rnn --bs 8 --lr 7e-5  --room-loss-weight $w ; done

# experiments for table 5
# baseline, mvcnn, dan, vsformer, hierartex, hicalm, with clip and mobile_clip, with lr 7e-5
l=7e-5
m=clip
echo "\n ======== HICALM ${m} ${l} ======== \n"
python run_main_HierArtEx_v2_add_gaming.py --generalist $m  --room_txt_agg rnn --room_vis_agg rnn --bs 8 --lr $l
python eval_main_HierArtEx_v2_add_gaming.py --generalist $m  --room_txt_agg rnn --room_vis_agg rnn --bs 8 --lr $l
echo "\n ======== HIERART ${m} ${l} ======== \n"
python run_main_HierArtEx_v2_add_gaming.py --generalist $m  --room_txt_agg rnn --room_vis_agg avg --bs 8 --lr $l  --room-loss-weight 0. --no-hiertxt
python eval_main_HierArtEx_v2_add_gaming.py --generalist $m  --room_txt_agg rnn --room_vis_agg avg --bs 8 --lr $l  --room-loss-weight 0. --no-hiertxt
#echo "\n ======== BASE (no artexp) ${m} ${l} ======== \n"
#python run_main_HierArtEx_v2_add_gaming.py --generalist $m  --room_txt_agg rnn --room_vis_agg avg --bs 8 --lr $l  --room-loss-weight 0. --no-hiertxt --no-hiervis --no-artexp
#python eval_main_HierArtEx_v2_add_gaming.py --generalist $m  --room_txt_agg rnn --room_vis_agg avg --bs 8 --lr $l  --room-loss-weight 0. --no-hiertxt --no-hiervis --no-artexp
echo "\n ======== BASE (with artexp) ${m} ${l} ======== \n"
python run_main_HierArtEx_v2_add_gaming.py --generalist $m  --room_txt_agg rnn --room_vis_agg avg --bs 8 --lr $l  --room-loss-weight 0. --no-hiertxt --no-hiervis
python eval_main_HierArtEx_v2_add_gaming.py --generalist $m  --room_txt_agg rnn --room_vis_agg avg --bs 8 --lr $l  --room-loss-weight 0. --no-hiertxt --no-hiervis
echo "\n ======== DAN ${m} ${l} ======== \n"
python run_main_HierArtEx_v2_add_gaming.py --generalist $m  --room_txt_agg rnn --bs 8 --lr $l --room-loss-weight 0. --no-hiertxt --other-method DAN --no-artexp 
python eval_main_HierArtEx_v2_add_gaming.py --generalist $m  --room_txt_agg rnn --bs 8 --lr $l --room-loss-weight 0. --no-hiertxt --other-method DAN --no-artexp 
echo "\n ======== VSFormer ${m} ${l} ======== \n" 
python run_main_HierArtEx_v2_add_gaming.py --generalist $m  --room_txt_agg rnn --bs 8 --lr $l --room-loss-weight 0. --no-hiertxt --other-method VSFormer --no-artexp
python eval_main_HierArtEx_v2_add_gaming.py --generalist $m  --room_txt_agg rnn --bs 8 --lr $l --room-loss-weight 0. --no-hiertxt --other-method VSFormer --no-artexp
echo "\n ======== MVCNN ${m} ${l} ======== \n"
python run_main_HierArtEx_v2_add_gaming.py --generalist $m  --room_txt_agg rnn --bs 8 --lr $l --room-loss-weight 0. --no-hiertxt --other-method MVCNN --no-artexp
python eval_main_HierArtEx_v2_add_gaming.py --generalist $m  --room_txt_agg rnn --bs 8 --lr $l --room-loss-weight 0. --no-hiertxt --other-method MVCNN --no-artexp

m=clip
echo "\n ======== BASE (with artexp) ${m} ${l} ======== \n"
python eval_main_HierArtEx_v2_add_gaming.py --generalist $m  --room_txt_agg rnn --room_vis_agg avg --bs 8 --lr $l  --room-loss-weight 0. --no-hiertxt --no-hiervis
#echo "\n ======== BASE (no artexp) ${m} ${l} ======== \n"
#python eval_main_HierArtEx_v2_add_gaming.py --generalist $m  --room_txt_agg rnn --room_vis_agg avg --bs 8 --lr $l  --room-loss-weight 0. --no-hiertxt --no-hiervis --no-artexp
echo "\n ======== MVCNN ${m} ${l} ======== \n"
python eval_main_HierArtEx_v2_add_gaming.py --generalist $m  --room_txt_agg rnn --bs 8 --lr $l --room-loss-weight 0. --no-hiertxt --other-method MVCNN --no-artexp
echo "\n ======== DAN ${m} ${l} ======== \n"
python eval_main_HierArtEx_v2_add_gaming.py --generalist $m  --room_txt_agg rnn --bs 8 --lr $l --room-loss-weight 0. --no-hiertxt --other-method DAN --no-artexp 
echo "\n ======== VSFormer ${m} ${l} ======== \n" 
python eval_main_HierArtEx_v2_add_gaming.py --generalist $m  --room_txt_agg rnn --bs 8 --lr $l --room-loss-weight 0. --no-hiertxt --other-method VSFormer --no-artexp
echo "\n ======== HIERART ${m} ${l} ======== \n"
python eval_main_HierArtEx_v2_add_gaming.py --generalist $m  --room_txt_agg rnn --room_vis_agg avg --bs 8 --lr $l  --room-loss-weight 0. --no-hiertxt
echo "\n ======== HICALM ${m} ${l} ======== \n"
python eval_main_HierArtEx_v2_add_gaming.py --generalist $m  --room_txt_agg rnn --room_vis_agg rnn --bs 8 --lr $l


m=mobile_clip
echo "\n ======== HICALM ${m} ${l} ======== \n"
python run_main_HierArtEx_v2_add_gaming.py --generalist $m  --room_txt_agg rnn --room_vis_agg rnn --bs 8 --lr $l
echo "\n ======== HIERART ${m} ${l} ======== \n"
python run_main_HierArtEx_v2_add_gaming.py --generalist $m  --room_txt_agg rnn --room_vis_agg avg --bs 8 --lr $l  --room-loss-weight 0. --no-hiertxt
echo "\n ======== BASE (no artexp) ${m} ${l} ======== \n"
python run_main_HierArtEx_v2_add_gaming.py --generalist $m  --room_txt_agg rnn --room_vis_agg avg --bs 8 --lr $l  --room-loss-weight 0. --no-hiertxt --no-hiervis --no-artexp
echo "\n ======== BASE (with artexp) ${m} ${l} ======== \n"
python run_main_HierArtEx_v2_add_gaming.py --generalist $m  --room_txt_agg rnn --room_vis_agg avg --bs 8 --lr $l  --room-loss-weight 0. --no-hiertxt --no-hiervis
echo "\n ======== DAN ${m} ${l} ======== \n"
python run_main_HierArtEx_v2_add_gaming.py --generalist $m  --room_txt_agg rnn --bs 8 --lr $l --room-loss-weight 0. --no-hiertxt --other-method DAN --no-artexp 
echo "\n ======== VSFormer ${m} ${l} ======== \n" 
python run_main_HierArtEx_v2_add_gaming.py --generalist $m  --room_txt_agg rnn --bs 8 --lr $l --room-loss-weight 0. --no-hiertxt --other-method VSFormer --no-artexp
echo "\n ======== MVCNN ${m} ${l} ======== \n"
python run_main_HierArtEx_v2_add_gaming.py --generalist $m  --room_txt_agg rnn --bs 8 --lr $l --room-loss-weight 0. --no-hiertxt --other-method MVCNN --no-artexp

echo "\n ======== BASE (with artexp) ${m} ${l} ======== \n"
python eval_main_HierArtEx_v2_add_gaming.py --generalist $m  --room_txt_agg rnn --room_vis_agg avg --bs 8 --lr $l  --room-loss-weight 0. --no-hiertxt --no-hiervis
#echo "\n ======== BASE (no artexp) ${m} ${l} ======== \n"
#python eval_main_HierArtEx_v2_add_gaming.py --generalist $m  --room_txt_agg rnn --room_vis_agg avg --bs 8 --lr $l  --room-loss-weight 0. --no-hiertxt --no-hiervis --no-artexp
echo "\n ======== MVCNN ${m} ${l} ======== \n"
python eval_main_HierArtEx_v2_add_gaming.py --generalist $m  --room_txt_agg rnn --bs 8 --lr $l --room-loss-weight 0. --no-hiertxt --other-method MVCNN --no-artexp
echo "\n ======== DAN ${m} ${l} ======== \n"
python eval_main_HierArtEx_v2_add_gaming.py --generalist $m  --room_txt_agg rnn --bs 8 --lr $l --room-loss-weight 0. --no-hiertxt --other-method DAN --no-artexp 
echo "\n ======== VSFormer ${m} ${l} ======== \n" 
python eval_main_HierArtEx_v2_add_gaming.py --generalist $m  --room_txt_agg rnn --bs 8 --lr $l --room-loss-weight 0. --no-hiertxt --other-method VSFormer --no-artexp
echo "\n ======== HIERART ${m} ${l} ======== \n"
python eval_main_HierArtEx_v2_add_gaming.py --generalist $m  --room_txt_agg rnn --room_vis_agg avg --bs 8 --lr $l  --room-loss-weight 0. --no-hiertxt
echo "\n ======== HICALM ${m} ${l} ======== \n"
python eval_main_HierArtEx_v2_add_gaming.py --generalist $m  --room_txt_agg rnn --room_vis_agg rnn --bs 8 --lr $l


# experiment for table 6 with user queries
# python eval_main_HierArtEx_v2_add_gaming.py --generalist clip  --room_txt_agg rnn --room_vis_agg avg --bs 8  --room-loss-weight 0. --no-hiertxt --skip-ndcg --eval-user-queries

m=mobile_clip
echo "\n ======== BASE (with artexp) ${m} ${l} ======== \n"
python eval_main_HierArtEx_v2_.py --generalist $m  --room_txt_agg rnn --room_vis_agg avg --lr $l  --room-loss-weight 0. --no-hiertxt --no-hiervis --skip-ndcg --eval-user-queries

