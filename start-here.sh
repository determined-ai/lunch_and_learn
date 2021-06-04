# The only package you will need to install is determined.
# We recommend you set it up in a virtual environment as below.
python3 -m venv ll_env
source ll_env/bin/activate
pip install determined

# Fill in the first and last name you use to register
# in lower case letters.
export FIRST_NAME=[fill_in]
export LAST_NAME=[fill_in]

# You will be given the determined cluster URL once the
# webinar starts.
export DET_MASTER=[fill_in]
# Once you get the determined cluster URL, you can login
# with your account.
det user login ${FIRST_NAME}_${LAST_NAME}
# password is blank initially.  you can change it by running
#det user change-password
