#
# add username : password pairs to the _users dict to control access to the Dash site 
# when someone navigates to the site, it will look like:  https://dash.plotly.com/authentication#basic-auth-example
# to logout a user will need to clear their browser's cache for this site
#
users = {
    # "user" : "password"
}

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import dash_auth

def Authenticate(app):
    if len(users) > 0:
        auth = dash_auth.BasicAuth(app, users)
        return auth