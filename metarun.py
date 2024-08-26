./run.sh train synthetic EDS_REFINE
./run.sh train dtu EDS_REFINE
./run.sh train bm EDS_REFINE
./run.sh train mipnerf EDS_REFINE

./run.sh train synthetic EDS_REFINE
./run.sh train dtu EDS_REFINE
./run.sh train bm EDS_REFINE
./run.sh train synthetic EDS_REFINE_MORELEVEL --network morelevels.json
./run.sh train dtu EDS_REFINE_MORELEVEL --network morelevels.json
./run.sh train bm EDS_REFINE_MORELEVEL --network morelevels.json


./run.sh train dtu LAP_vanishinglap_from0d02
./run.sh train mipnerf LAP_vanishinglap_from0d02

./run.sh train_screen dtu COLOR



laplacian_weight_decay_min=0.001 ./run.sh train_screen custom TMP0.001
laplacian_weight_decay_min=0.0005 ./run.sh train_screen custom TMP0.0005
laplacian_weight_decay_min=0.0002 ./run.sh train_screen custom TMP0.0002
laplacian_weight_decay_min=0.0001 ./run.sh train_screen custom TMP0.0001