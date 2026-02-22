c = get_config()

# Use system (PAM) authentication
c.JupyterHub.authenticator_class = "jupyterhub.auth.PAMAuthenticator"

# JupyterHub 5+ requires an explicit allow rule
c.Authenticator.allow_all = True

# Optional stricter alternative (instead of allow_all):
# c.Authenticator.allowed_users = {"maxime"}
