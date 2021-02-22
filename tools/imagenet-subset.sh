#!/bin/bash
#
# this script takes the 1000 ImageNet classes from ILSVRC12
# and condenses it down to a subset of 12 classes in DIGITS 
# directory format using symbolic links to the ILSVRC12 data.
#
#   usage:  ./imagenet-subset.sh <absolute-path-to-ilsvrc12> <output-path>
#

DATASET_PATH=$1
OUTPUT_PATH=$2


mkdir $OUTPUT_PATH/ball

cp -r $DATASET_PATH/n02799071 $OUTPUT_PATH/ball/baseball
cp -r $DATASET_PATH/n02802426 $OUTPUT_PATH/ball/basketball
cp -r $DATASET_PATH/n03134739 $OUTPUT_PATH/ball/croquet_ball
cp -r $DATASET_PATH/n04118538 $OUTPUT_PATH/ball/rugby_ball
cp -r $DATASET_PATH/n04254680 $OUTPUT_PATH/ball/soccer_ball
cp -r $DATASET_PATH/n04409515 $OUTPUT_PATH/ball/tennis_ball
cp -r $DATASET_PATH/n04540053 $OUTPUT_PATH/ball/volleyball

mkdir $OUTPUT_PATH/bear

cp -r $DATASET_PATH/n02132136 $OUTPUT_PATH/bear/brown_bear
cp -r $DATASET_PATH/n02133161 $OUTPUT_PATH/bear/black_bear
cp -r $DATASET_PATH/n02134084 $OUTPUT_PATH/bear/polar_bear
cp -r $DATASET_PATH/n02134418 $OUTPUT_PATH/bear/sloth_bear

mkdir $OUTPUT_PATH/bike

cp -r $DATASET_PATH/n02835271 $OUTPUT_PATH/bike/tandem_bicycle
cp -r $DATASET_PATH/n03791053 $OUTPUT_PATH/bike/scooter
cp -r $DATASET_PATH/n03792782 $OUTPUT_PATH/bike/mountain_bike



mkdir $OUTPUT_PATH/bird

cp -r $DATASET_PATH/n01514668 $OUTPUT_PATH/bird/rooster
cp -r $DATASET_PATH/n01514859 $OUTPUT_PATH/bird/hen
cp -r $DATASET_PATH/n01530575 $OUTPUT_PATH/bird/brambling
cp -r $DATASET_PATH/n01531178 $OUTPUT_PATH/bird/goldfinch
cp -r $DATASET_PATH/n01532829 $OUTPUT_PATH/bird/finch
cp -r $DATASET_PATH/n01534433 $OUTPUT_PATH/bird/junco
cp -r $DATASET_PATH/n01537544 $OUTPUT_PATH/bird/indigo
cp -r $DATASET_PATH/n01558993 $OUTPUT_PATH/bird/robin
cp -r $DATASET_PATH/n01560419 $OUTPUT_PATH/bird/bulbul
cp -r $DATASET_PATH/n01580077 $OUTPUT_PATH/bird/jay
cp -r $DATASET_PATH/n01582220 $OUTPUT_PATH/bird/magpie
cp -r $DATASET_PATH/n01592084 $OUTPUT_PATH/bird/chickadee
cp -r $DATASET_PATH/n01601694 $OUTPUT_PATH/bird/dipper
cp -r $DATASET_PATH/n01608432 $OUTPUT_PATH/bird/kite
cp -r $DATASET_PATH/n01614925 $OUTPUT_PATH/bird/bald_eagle
cp -r $DATASET_PATH/n01616318 $OUTPUT_PATH/bird/vulture
cp -r $DATASET_PATH/n01622779 $OUTPUT_PATH/bird/grey_owl


mkdir $OUTPUT_PATH/bottle

cp -r $DATASET_PATH/n02815834 $OUTPUT_PATH/bottle/beaker
cp -r $DATASET_PATH/n02823428 $OUTPUT_PATH/bottle/beer_bottle
cp -r $DATASET_PATH/n02823750 $OUTPUT_PATH/bottle/beer_glass
cp -r $DATASET_PATH/n03063599 $OUTPUT_PATH/bottle/coffee_mug
cp -r $DATASET_PATH/n04557648 $OUTPUT_PATH/bottle/water_bottle
cp -r $DATASET_PATH/n04560804 $OUTPUT_PATH/bottle/water_jug
cp -r $DATASET_PATH/n04591713 $OUTPUT_PATH/bottle/wine_bottle


mkdir $OUTPUT_PATH/cat

cp -r $DATASET_PATH/n02123045 $OUTPUT_PATH/cat/tabby
cp -r $DATASET_PATH/n02123159 $OUTPUT_PATH/cat/tiger_cat
cp -r $DATASET_PATH/n02123394 $OUTPUT_PATH/cat/Persian
cp -r $DATASET_PATH/n02123597 $OUTPUT_PATH/cat/Siamese
cp -r $DATASET_PATH/n02124075 $OUTPUT_PATH/cat/Egyptian
cp -r $DATASET_PATH/n02125311 $OUTPUT_PATH/cat/cougar
cp -r $DATASET_PATH/n02127052 $OUTPUT_PATH/cat/lynx
cp -r $DATASET_PATH/n02128385 $OUTPUT_PATH/cat/leopard
cp -r $DATASET_PATH/n02128757 $OUTPUT_PATH/cat/snow_leopard
cp -r $DATASET_PATH/n02128925 $OUTPUT_PATH/cat/jaguar
cp -r $DATASET_PATH/n02129165 $OUTPUT_PATH/cat/lion
cp -r $DATASET_PATH/n02129604 $OUTPUT_PATH/cat/tiger
cp -r $DATASET_PATH/n02130308 $OUTPUT_PATH/cat/cheetah


mkdir $OUTPUT_PATH/dog

cp -r $DATASET_PATH/n02085620 $OUTPUT_PATH/dog/Chihuahua
cp -r $DATASET_PATH/n02085782 $OUTPUT_PATH/dog/Japanese_spaniel
cp -r $DATASET_PATH/n02085936 $OUTPUT_PATH/dog/Maltese
cp -r $DATASET_PATH/n02086079 $OUTPUT_PATH/dog/Pekinese
cp -r $DATASET_PATH/n02086240 $OUTPUT_PATH/dog/Shih_Tzu
cp -r $DATASET_PATH/n02086646 $OUTPUT_PATH/dog/Blenheim_spaniel
cp -r $DATASET_PATH/n02086910 $OUTPUT_PATH/dog/papillon
cp -r $DATASET_PATH/n02087046 $OUTPUT_PATH/dog/toy_terrier
cp -r $DATASET_PATH/n02087394 $OUTPUT_PATH/dog/Rhodesian_ridgeback
cp -r $DATASET_PATH/n02088094 $OUTPUT_PATH/dog/Afghan_hound
cp -r $DATASET_PATH/n02088238 $OUTPUT_PATH/dog/basset_hound
cp -r $DATASET_PATH/n02088364 $OUTPUT_PATH/dog/beagle
cp -r $DATASET_PATH/n02088466 $OUTPUT_PATH/dog/bloodhound
cp -r $DATASET_PATH/n02088632 $OUTPUT_PATH/dog/bluetick
cp -r $DATASET_PATH/n02089078 $OUTPUT_PATH/dog/coonhound
cp -r $DATASET_PATH/n02089867 $OUTPUT_PATH/dog/Walker_foxhound
cp -r $DATASET_PATH/n02089973 $OUTPUT_PATH/dog/English_foxhound
cp -r $DATASET_PATH/n02090379 $OUTPUT_PATH/dog/redbone
cp -r $DATASET_PATH/n02090622 $OUTPUT_PATH/dog/borzoi
cp -r $DATASET_PATH/n02090721 $OUTPUT_PATH/dog/Irish_wolfhound
cp -r $DATASET_PATH/n02091032 $OUTPUT_PATH/dog/Italian_greyhound
cp -r $DATASET_PATH/n02091134 $OUTPUT_PATH/dog/whippet
cp -r $DATASET_PATH/n02091244 $OUTPUT_PATH/dog/Ibizan_hound
cp -r $DATASET_PATH/n02091467 $OUTPUT_PATH/dog/Norwegian_elkhound
cp -r $DATASET_PATH/n02091635 $OUTPUT_PATH/dog/otter_hound
cp -r $DATASET_PATH/n02091831 $OUTPUT_PATH/dog/Saluki
cp -r $DATASET_PATH/n02092002 $OUTPUT_PATH/dog/Scottish_deerhound
cp -r $DATASET_PATH/n02092339 $OUTPUT_PATH/dog/Weimaraner
cp -r $DATASET_PATH/n02093256 $OUTPUT_PATH/dog/Staffordshire_bullterrier
cp -r $DATASET_PATH/n02093428 $OUTPUT_PATH/dog/American_Staffordshire_terrier
cp -r $DATASET_PATH/n02093647 $OUTPUT_PATH/dog/Bedlington_terrier
cp -r $DATASET_PATH/n02093754 $OUTPUT_PATH/dog/Border_terrier
cp -r $DATASET_PATH/n02093859 $OUTPUT_PATH/dog/Kerry_blue_terrier
cp -r $DATASET_PATH/n02093991 $OUTPUT_PATH/dog/Irish_terrier
cp -r $DATASET_PATH/n02094114 $OUTPUT_PATH/dog/Norfolk_terrier
cp -r $DATASET_PATH/n02094258 $OUTPUT_PATH/dog/Norwich_terrier
cp -r $DATASET_PATH/n02094433 $OUTPUT_PATH/dog/Yorkshire_terrier
cp -r $DATASET_PATH/n02095314 $OUTPUT_PATH/dog/wirehaired_fox_terrier
cp -r $DATASET_PATH/n02095570 $OUTPUT_PATH/dog/Lakeland_terrier
cp -r $DATASET_PATH/n02095889 $OUTPUT_PATH/dog/Sealyham_terrier
cp -r $DATASET_PATH/n02096051 $OUTPUT_PATH/dog/Airedale
cp -r $DATASET_PATH/n02096177 $OUTPUT_PATH/dog/cairn
cp -r $DATASET_PATH/n02096294 $OUTPUT_PATH/dog/Australian_terrier
cp -r $DATASET_PATH/n02096437 $OUTPUT_PATH/dog/Dandie_Dinmont_terrier
cp -r $DATASET_PATH/n02096585 $OUTPUT_PATH/dog/Boston_terrier
cp -r $DATASET_PATH/n02097047 $OUTPUT_PATH/dog/miniature_schnauzer
cp -r $DATASET_PATH/n02097130 $OUTPUT_PATH/dog/giant_schnauzer
cp -r $DATASET_PATH/n02097209 $OUTPUT_PATH/dog/standard_schnauzer
cp -r $DATASET_PATH/n02097298 $OUTPUT_PATH/dog/Scottie
cp -r $DATASET_PATH/n02097474 $OUTPUT_PATH/dog/Tibetan_terrier
cp -r $DATASET_PATH/n02097658 $OUTPUT_PATH/dog/silky_terrier
cp -r $DATASET_PATH/n02098105 $OUTPUT_PATH/dog/softcoated_wheaten_terrier
cp -r $DATASET_PATH/n02098286 $OUTPUT_PATH/dog/West_Highland_white_terrier
cp -r $DATASET_PATH/n02098413 $OUTPUT_PATH/dog/Lhasa
cp -r $DATASET_PATH/n02099267 $OUTPUT_PATH/dog/flat_coated_retriever
cp -r $DATASET_PATH/n02099429 $OUTPUT_PATH/dog/curly_coated_retriever
cp -r $DATASET_PATH/n02099601 $OUTPUT_PATH/dog/golden_retriever
cp -r $DATASET_PATH/n02099712 $OUTPUT_PATH/dog/Labrador_retriever
cp -r $DATASET_PATH/n02099849 $OUTPUT_PATH/dog/Chesapeake_Bay_retriever
cp -r $DATASET_PATH/n02100236 $OUTPUT_PATH/dog/German_short_haired_pointer
cp -r $DATASET_PATH/n02100583 $OUTPUT_PATH/dog/Hungarian_pointer
cp -r $DATASET_PATH/n02100735 $OUTPUT_PATH/dog/English_setter
cp -r $DATASET_PATH/n02100877 $OUTPUT_PATH/dog/Irish_setter
cp -r $DATASET_PATH/n02101006 $OUTPUT_PATH/dog/Gordon_setter
cp -r $DATASET_PATH/n02101388 $OUTPUT_PATH/dog/Brittany_spaniel
cp -r $DATASET_PATH/n02101556 $OUTPUT_PATH/dog/clumber_spaniel
cp -r $DATASET_PATH/n02102040 $OUTPUT_PATH/dog/English_springer
cp -r $DATASET_PATH/n02102177 $OUTPUT_PATH/dog/Welsh_springer_spaniel
cp -r $DATASET_PATH/n02102318 $OUTPUT_PATH/dog/cocker_spaniel
cp -r $DATASET_PATH/n02102480 $OUTPUT_PATH/dog/Sussex_spaniel
cp -r $DATASET_PATH/n02102973 $OUTPUT_PATH/dog/Irish_water_spaniel
cp -r $DATASET_PATH/n02104029 $OUTPUT_PATH/dog/kuvasz
cp -r $DATASET_PATH/n02104365 $OUTPUT_PATH/dog/schipperke
cp -r $DATASET_PATH/n02105056 $OUTPUT_PATH/dog/groenendael
cp -r $DATASET_PATH/n02105162 $OUTPUT_PATH/dog/malinois
cp -r $DATASET_PATH/n02105251 $OUTPUT_PATH/dog/briard
cp -r $DATASET_PATH/n02105412 $OUTPUT_PATH/dog/kelpie
cp -r $DATASET_PATH/n02105505 $OUTPUT_PATH/dog/komondor
cp -r $DATASET_PATH/n02105641 $OUTPUT_PATH/dog/Old_English_sheepdog
cp -r $DATASET_PATH/n02105855 $OUTPUT_PATH/dog/Shetland_sheepdog
cp -r $DATASET_PATH/n02106030 $OUTPUT_PATH/dog/collie
cp -r $DATASET_PATH/n02106166 $OUTPUT_PATH/dog/Border_collie
cp -r $DATASET_PATH/n02106382 $OUTPUT_PATH/dog/Bouvier
cp -r $DATASET_PATH/n02106550 $OUTPUT_PATH/dog/Rottweiler
cp -r $DATASET_PATH/n02106662 $OUTPUT_PATH/dog/German_shepherd
cp -r $DATASET_PATH/n02107142 $OUTPUT_PATH/dog/Doberman
cp -r $DATASET_PATH/n02107312 $OUTPUT_PATH/dog/miniature_pinscher
cp -r $DATASET_PATH/n02107574 $OUTPUT_PATH/dog/Greater_Swiss_Mountain_dog
cp -r $DATASET_PATH/n02107683 $OUTPUT_PATH/dog/Bernese_mountain_dog
cp -r $DATASET_PATH/n02107908 $OUTPUT_PATH/dog/Appenzeller
cp -r $DATASET_PATH/n02108000 $OUTPUT_PATH/dog/EntleBucher
cp -r $DATASET_PATH/n02108089 $OUTPUT_PATH/dog/boxer
cp -r $DATASET_PATH/n02108422 $OUTPUT_PATH/dog/bull_mastiff
cp -r $DATASET_PATH/n02108551 $OUTPUT_PATH/dog/Tibetan_mastiff
cp -r $DATASET_PATH/n02108915 $OUTPUT_PATH/dog/French_bulldog
cp -r $DATASET_PATH/n02109047 $OUTPUT_PATH/dog/Great_Dane
cp -r $DATASET_PATH/n02109525 $OUTPUT_PATH/dog/Saint_Bernard
cp -r $DATASET_PATH/n02109961 $OUTPUT_PATH/dog/husky
cp -r $DATASET_PATH/n02110063 $OUTPUT_PATH/dog/malamute
cp -r $DATASET_PATH/n02110185 $OUTPUT_PATH/dog/Siberian_husky
cp -r $DATASET_PATH/n02110341 $OUTPUT_PATH/dog/dalmatian
cp -r $DATASET_PATH/n02110627 $OUTPUT_PATH/dog/affenpinscher
cp -r $DATASET_PATH/n02110806 $OUTPUT_PATH/dog/basenji
cp -r $DATASET_PATH/n02110958 $OUTPUT_PATH/dog/pug
cp -r $DATASET_PATH/n02111129 $OUTPUT_PATH/dog/Leonberg
cp -r $DATASET_PATH/n02111277 $OUTPUT_PATH/dog/Newfoundland
cp -r $DATASET_PATH/n02111500 $OUTPUT_PATH/dog/Great_Pyrenees
cp -r $DATASET_PATH/n02111889 $OUTPUT_PATH/dog/Samoyed
cp -r $DATASET_PATH/n02112018 $OUTPUT_PATH/dog/Pomeranian
cp -r $DATASET_PATH/n02112137 $OUTPUT_PATH/dog/chow
cp -r $DATASET_PATH/n02112350 $OUTPUT_PATH/dog/keeshond
cp -r $DATASET_PATH/n02112706 $OUTPUT_PATH/dog/Brabancon_griffon
cp -r $DATASET_PATH/n02113023 $OUTPUT_PATH/dog/Pembroke
cp -r $DATASET_PATH/n02113186 $OUTPUT_PATH/dog/Cardigan
cp -r $DATASET_PATH/n02113624 $OUTPUT_PATH/dog/toy_poodle
cp -r $DATASET_PATH/n02113712 $OUTPUT_PATH/dog/miniature_poodle
cp -r $DATASET_PATH/n02113799 $OUTPUT_PATH/dog/standard_poodle
cp -r $DATASET_PATH/n02113978 $OUTPUT_PATH/dog/Mexican_hairless
cp -r $DATASET_PATH/n02114367 $OUTPUT_PATH/dog/grey_wolf
cp -r $DATASET_PATH/n02114548 $OUTPUT_PATH/dog/white_wolf
cp -r $DATASET_PATH/n02114712 $OUTPUT_PATH/dog/red_wolf
cp -r $DATASET_PATH/n02114855 $OUTPUT_PATH/dog/coyote
cp -r $DATASET_PATH/n02115641 $OUTPUT_PATH/dog/dingo

mkdir $OUTPUT_PATH/fish

cp -r $DATASET_PATH/n01440764 $OUTPUT_PATH/fish/tench
cp -r $DATASET_PATH/n01443537 $OUTPUT_PATH/fish/goldfish
cp -r $DATASET_PATH/n01484850 $OUTPUT_PATH/fish/great_white
cp -r $DATASET_PATH/n01491361 $OUTPUT_PATH/fish/tiger_shark
cp -r $DATASET_PATH/n01498041 $OUTPUT_PATH/fish/stingray


mkdir $OUTPUT_PATH/fruit

cp -r $DATASET_PATH/n07720875 $OUTPUT_PATH/fruit/bell_pepper
cp -r $DATASET_PATH/n07742313 $OUTPUT_PATH/fruit/Granny_Smith
cp -r $DATASET_PATH/n07745940 $OUTPUT_PATH/fruit/strawberry
cp -r $DATASET_PATH/n07747607 $OUTPUT_PATH/fruit/orange
cp -r $DATASET_PATH/n07749582 $OUTPUT_PATH/fruit/lemon
cp -r $DATASET_PATH/n07753113 $OUTPUT_PATH/fruit/fig
cp -r $DATASET_PATH/n07753275 $OUTPUT_PATH/fruit/pineapple
cp -r $DATASET_PATH/n07753592 $OUTPUT_PATH/fruit/banana
cp -r $DATASET_PATH/n07754684 $OUTPUT_PATH/fruit/jackfruit
cp -r $DATASET_PATH/n07760859 $OUTPUT_PATH/fruit/custard_apple
cp -r $DATASET_PATH/n07768694 $OUTPUT_PATH/fruit/pomegranate


mkdir $OUTPUT_PATH/turtle

cp -r $DATASET_PATH/n01664065 $OUTPUT_PATH/turtle/loggerhead
cp -r $DATASET_PATH/n01665541 $OUTPUT_PATH/turtle/leatherback
cp -r $DATASET_PATH/n01667114 $OUTPUT_PATH/turtle/mud
cp -r $DATASET_PATH/n01667778 $OUTPUT_PATH/turtle/terrapin
cp -r $DATASET_PATH/n01669191 $OUTPUT_PATH/turtle/box

mkdir $OUTPUT_PATH/vehicle

cp -r $DATASET_PATH/n02701002 $OUTPUT_PATH/vehicle/ambulance
cp -r $DATASET_PATH/n02930766 $OUTPUT_PATH/vehicle/taxi
cp -r $DATASET_PATH/n03345487 $OUTPUT_PATH/vehicle/fire_engine
cp -r $DATASET_PATH/n03417042 $OUTPUT_PATH/vehicle/garbage_truck
cp -r $DATASET_PATH/n03670208 $OUTPUT_PATH/vehicle/limousine
cp -r $DATASET_PATH/n03769881 $OUTPUT_PATH/vehicle/minibus
cp -r $DATASET_PATH/n03930630 $OUTPUT_PATH/vehicle/pickup_truck
cp -r $DATASET_PATH/n03977966 $OUTPUT_PATH/vehicle/police_wagon
cp -r $DATASET_PATH/n04065272 $OUTPUT_PATH/vehicle/rv
cp -r $DATASET_PATH/n03770679 $OUTPUT_PATH/vehicle/minivan
cp -r $DATASET_PATH/n04285008 $OUTPUT_PATH/vehicle/sports_car
cp -r $DATASET_PATH/n04461696 $OUTPUT_PATH/vehicle/tow_truck
cp -r $DATASET_PATH/n04467665 $OUTPUT_PATH/vehicle/tractor_trailer
cp -r $DATASET_PATH/n04487081 $OUTPUT_PATH/vehicle/trolleybus

mkdir $OUTPUT_PATH/sign

cp -r $DATASET_PATH/n06794110 $OUTPUT_PATH/sign/sign_street_sign
cp -r $DATASET_PATH/n06874185 $OUTPUT_PATH/sign/sign_traffic_light


