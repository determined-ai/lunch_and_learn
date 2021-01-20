# Fill in the first and last name you use to register
# in lower case letters.
export FIRST_NAME=[FILL_IN]
export LAST_NAME=[FILL_IN]

# You will be given the determined cluster URL once the
# webinar starts.
export DET_MASTER=[FILL_IN]

# The only package you will need to install is the determined-cli.
# We recommend you set it up in a virtual environment as below.
python3 -m venv lunch_and_learn
source lunch_and_learn/bin/activate
pip install determined-cli

# Once you get the determined cluster URL, you can login 
# with your account.
det user login ${FIRST_NAME}_${LAST_NAME}
# password is blank initially.  you can change it by running
det user change-password
