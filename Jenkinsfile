// Jenkins Pipeline for Weather Agent SAM deployment.
// This pipeline validates/builds on every branch and deploys from `main`.

pipeline {
  agent any

  environment {
    APP_DIR = 'ai-agents/weather'
    TEMPLATE_FILE = 'template.yaml'
    STACK_NAME = 'wweather'
    AWS_REGION = 'ap-south-1'
  }

  options {
    timestamps()
    disableConcurrentBuilds()
  }

  stages {
    stage('Checkout') {
      steps {
        checkout scm
      }
    }

    stage('Tool Check') {
      steps {
        sh '''
          set -e
          cd "$APP_DIR"
          node --version
          sam --version
          aws --version
        '''
      }
    }

    stage('Validate') {
      steps {
        sh '''
          set -e
          cd "$APP_DIR"
          sam validate -t "$TEMPLATE_FILE"
        '''
      }
    }

    stage('Build') {
      steps {
        sh '''
          set -e
          cd "$APP_DIR"
          sam build -t "$TEMPLATE_FILE"
        '''
      }
    }

    stage('Deploy (main only)') {
      when {
        branch 'main'
      }
      steps {
        withCredentials([
          [$class: 'AmazonWebServicesCredentialsBinding', credentialsId: 'aws-ci-user'],
          string(credentialsId: 'gemini-api-key', variable: 'GEMINI_API_KEY')
        ]) {
          sh '''
            set -e
            cd "$APP_DIR"

            sam deploy \
              -t "$TEMPLATE_FILE" \
              --stack-name "$STACK_NAME" \
              --region "$AWS_REGION" \
              --capabilities CAPABILITY_IAM \
              --resolve-s3 \
              --parameter-overrides "GeminiApiKey=$GEMINI_API_KEY" \
              --no-confirm-changeset \
              --no-fail-on-empty-changeset
          '''
        }
      }
    }
  }

  post {
    always {
      archiveArtifacts artifacts: 'ai-agents/weather/.aws-sam/**/*', allowEmptyArchive: true
    }
  }
}
