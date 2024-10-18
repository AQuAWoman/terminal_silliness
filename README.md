# terminal_silliness

Just a simple image convolution utility to create text-based art. Probably overdone at this point but this is my take. Enjoy!

Prerequisites (may be incomplete, please add pull request if there's more apt dependencies required):

- Pip
`$ sudo apt install pip`

- Python 3
`$ sudo apt install python3`

Installation:

- Create virtual environment

`$ python3 -m venv /path/to/venv/`

- Enter virtual environment

`$ source /path/to/venv/bin/activate`

- Install dependencies

`$ pip install -r ./requirements.txt`


Usage:

`$ python3 terminal_silliness.py \<image path or url> [target width (characters)] [FG[-BG][-I][-B][-N]], ...`

FG and BG are numbers, I, B, and N denote italics, bold, and negation respectively.
