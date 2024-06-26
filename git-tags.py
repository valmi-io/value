import os
connectors = {
    "destination-webhook": "0.1.17",
    "destination-stripe": "0.1.9",
    "destination-slack": "0.1.9",
    "destination-hubspot": "0.1.9",
    "destination-google-sheets": "0.1.10",
    "destination-google-ads": "0.1.9",
    "destination-gong": "0.1.9",
    "destination-facebook-ads": "0.1.9",
    "destination-customer-io": "0.1.9",
    "destination-android-push-notifications": "0.1.10",

    "source-redshift": "0.1.13",
    "source-snowflake": "0.1.13",
    "source-postgres": "0.1.15"
}

valmi_core = "0.1.26"

valmi_app_backend = "0.1.38"

valmi_app = "0.1.41"

valmi_dagster = "0.1.5"


def tag_and_push(tag_name, message):
    os.system("git tag -a %s -m '%s'" % (tag_name, message))
    os.system("git push origin %s" % tag_name)


def tag_valmi_app_backend():
    tag_name = 'valmi-app-backend/%s' % valmi_app_backend
    os.system("cd valmi-app-backend; git tag -a %s -m '%s'" % (tag_name, 'creating tag %s' % tag_name))
    os.system("cd valmi-app-backend; git push origin %s" % tag_name)


def tag_valmi_app():
    tag_name = 'valmi-app/%s' % valmi_app
    os.system("cd valmi-app; git tag -a %s -m '%s'" % (tag_name, 'creating tag %s' % tag_name))
    os.system("cd valmi-app; git push origin %s" % tag_name)


def tag_dagster():
    tag_name = 'valmi-dagster/%s' % valmi_dagster
    os.system("git tag -a %s -m '%s'" % (tag_name, 'creating tag %s' % tag_name))
    os.system("git push origin %s" % tag_name)


for k, v in connectors.items():
    print("Tagging connector %s:%s" % (k, v))
    tag_and_push("valmi-connector/%s/%s" % (k, v), "Tagging %s:%s" % (k, v))


tag_and_push("valmi-activation/%s" % valmi_core, "Tagging valmi-activation/%s" % valmi_core)
tag_valmi_app()
tag_valmi_app_backend()
tag_dagster()
