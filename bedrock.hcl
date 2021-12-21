// Refer to https://docs.basis-ai.com/getting-started/writing-files/bedrock.hcl for more details.
version = "1.0"

batch_score {
    step "detect" {
        image = "quay.io/basisai/python-cuda:3.9.6-11.0"
        install = [
            "pip install -U pip",
            "pip install awscli",
            "pip install -r requirements.txt",
        ]
        script = [{sh = [
            "scripts/detect.sh",
        ]}]
        resources {
            cpu = "3"
            memory = "14G"
            gpu = "1"
        }
    }

    parameters {
        TEMP_BUCKET = "span-staging-temp-data"
    }
}
