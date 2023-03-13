#
# Networking config options can get set in a few ways, in increasing order of precedence:
#
#   1. the hardcoded default options (see 'config' dict below)
#   2. the json config file (by default this is data/config.json)
#   3. some environment variables:
#         $DASH_CONFIG_FILE
#         $SSH_CERT
#         $SSH_KEY
#         $STUN_SERVER
#   4. command-line arguments that some apps implement
#
import os
import json
import pprint
import mergedeep

# This config file gets loaded on start-up, or the defaults written to.
# It can also be set with the $DASH_CONFIG_FILE environment variable.
DASH_CONFIG_FILE=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/config.json')
DASH_CONFIG_FILE=os.getenv('DASH_CONFIG_FILE', DASH_CONFIG_FILE)


# These are the default config options that are used to populate DASH_CONFIG_FILE
# if it doesn't already exist. These shouldn't really be changed here because
# the config file will automatically be created and will then override these. 
config={
    'dash' : {                          # frontend webserver options  (consider renaming 'dash' to 'app')
        'title' : 'Hello AI World',     # title of the dash app (used in browser title bar and navbar)
        'host' : '0.0.0.0',             # hostname/IP of the frontend webserver (ignored by gunicorn)
        'port' : 8050,                  # port used for the frontend webserver (ignored by gunicorn)
        'refresh' : 2500,               # the interval at which the server state is refreshed
        'users' : {                     # to enable basic authentication logins, add username/password pairs here
            # 'username' : 'password',
        },
    },

    'server' : {                        # backend media server options
        'name' : 'server-backend',      # name of the backend server process (also used for logging)
        'host' : '0.0.0.0',             # hostname/IP of the backend server to bind/connect to
        'rest_port' : 49565,            # port used for JSON REST API
        'webrtc_port' : 8554,          # port used for WebRTC server
        'ssl_cert' : None,              # path to PEM-encoded SSL/TLS certificate file for enabling HTTPS
        'ssl_key' : None,               # path to PEM-encoded SSL/TLS key file for enabling HTTPS
        'stun_server' : None,           # override the default WebRTC STUN server (stun.l.google.com:19302)
    }
}


# Read a config file, or create it with the defaults if it doesn't exist.
# This automatically gets called to load $DASH_CONFIG_FILE on start.
# Returns the loaded configuration, or the default config if the file didn't exist
# TODO add error checking / exception handling in case of corrupt file
def load_config(path, save_defaults=True, set_global=True):
    #global config
    if os.path.exists(path):
        conf = config if set_global else {}
        with open(path) as file:
            mergedeep.merge(conf, json.load(file))  # merge nested dicts (config file replaces defaults)
        return conf
    elif save_defaults:
        with open(path, 'w') as file:
            json.dump(config, file, indent=4)
            print(f"wrote default config file to {path}")
    return config
    
# load the default configuration file
load_config(DASH_CONFIG_FILE) 

# check for some environment variables
for env in ['SSL_CERT', 'SSL_KEY', 'STUN_SERVER']:
    config['server'][env.lower()] = os.getenv(env, config['server'][env.lower()])

# log the configuration
def print_config(conf=config, prefix="# NETWORKING CONFIGURATION", indent=2):
    if prefix:
        print(prefix)
    pprint.pprint(config, indent=indent)   # sort_dicts=False (only on Python 3.8+)

