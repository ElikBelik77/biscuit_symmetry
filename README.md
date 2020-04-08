
# Biscuit Symmetry / Bilateral Symmetry
### COVID-19 Project #3
This project was inspired when I was wondering if my I can programmatically detect if my biscuits have bilateral symmetry.

## Description
This project can detect and mark bilateral symmetry using the algorithm from (https://link.springer.com/content/pdf/10.1007%2F11744047_39.pdf)[this paper]

## Run
To run the script, use the following command:
'''
python symmetry_detection.py --source image.png --out marked_image.png
'''
You can also see intermidate stages of the algorithm by using the flags:
'''
--save_feature_points
--save_hexbin
--save_matchpoints
'''

