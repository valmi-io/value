<p align="center">
  <a href="https://valmi.io"><img width="400" src="https://blog.valmi.io/content/images/2023/06/valmilogo-1.png" alt="valmi.io"></a>
</p>

<p align="center">test
  <b>
    <a href="https://www.valmi.io">Website</a>
    ·
    <a href="https://www.valmi.io/slack">Slack Community</a>
    ·
    <a href="https://www.valmi.io/docs/">Documentation</a>
    ·
    <a href="https://www.valmi.io/blog/">Blog</a>
  </b>
</p>

<p align="center">
    <em> <a href="https://valmi.io">valmi.io</a> is the open-source Reverse-ETL (data activation) platform to load data from warehouses into SaaS platforms, Webhook Apis etc.</em>
</p>
<p align="center">
<a href="https://github.com/valmi-io/valmi-activation/stargazers/" target="_blank">
    <img src="https://img.shields.io/github/stars/valmi-io/valmi-activation?style=social&label=Star&maxAge=10000" alt="Test">
</a>
<a href="https://github.com/valmi-io/valmi-activation/blob/main/LICENSE.md" target="_blank">
    <img src="https://img.shields.io/static/v1?label=license&message=MIT&color=white" alt="License">
</a>
</p>

  - ## See it in [Action](https://demo.valmi.io/spaces/a9195c50-60ca-4692-8f03-5a486ee9f270/syncs/d69cf9f9-0e20-4e2c-a683-2649404f52ed/runs)
    Quickly witness a One Million (1M) Rows Sync at [Live Sync](https://demo.valmi.io/spaces/a9195c50-60ca-4692-8f03-5a486ee9f270/syncs/d69cf9f9-0e20-4e2c-a683-2649404f52ed/runs).

  - ### We'd love to discuss your use-case and the problem you are trying to solve, if you are up for a 15-20 min call, please block a time on  [Co-Founder Raj Varkala's calendar](https://calendly.com/raj-varkala/30min)

    <div align="center">
    
     [![valmi-activation](https://github.com/valmi-io/valmi-activation/actions/workflows/valmi-activation-docker-image-action.yml/badge.svg)](https://github.com/valmi-io/valmi-activation/actions/workflows/valmi-activation-docker-image-action.yml) [![valmi-connectors](https://github.com/valmi-io/valmi-activation/actions/workflows/valmi-connectors-docker-image-action.yml/badge.svg)](https://github.com/valmi-io/valmi-activation/actions/workflows/valmi-connectors-docker-image-action.yml) [![valmi-dagster](https://github.com/valmi-io/valmi-activation/actions/workflows/valmi-dagster-docker-image-action.yml/badge.svg)](https://github.com/valmi-io/valmi-activation/actions/workflows/valmi-dagster-docker-image-action.yml) 
    <br/>
    [![valmi-repo](https://github.com/valmi-io/valmi-activation/actions/workflows/valmi-repo-docker-image-action.yml/badge.svg)](https://github.com/valmi-io/valmi-activation/actions/workflows/valmi-repo-docker-image-action.yml) [![valmi-app-backend](https://github.com/valmi-io/valmi-app-backend/actions/workflows/valmi-app-backend-docker-image-action.yml/badge.svg)](https://github.com/valmi-io/valmi-app-backend/actions/workflows/valmi-app-backend-docker-image-action.yml) [![valmi-app](https://github.com/valmi-io/valmi-app/actions/workflows/valmi-app-docker-image-action.yml/badge.svg)](https://github.com/valmi-io/valmi-app/actions/workflows/valmi-app-docker-image-action.yml)
    
      <!---
    <a href="/../../issues?q=is%3Aopen+is%3Aissue"> <img alt="GitHub issues" src="https://img.shields.io/github/issues-raw/valmi-io/valmi-activation?color=%23238636"></a> <a href="/../../issues?q=is%3Aissue+is%3Aclosed"> <img alt="GitHub closed issues" src="https://img.shields.io/github/issues-closed-raw/valmi-io/valmi-activation?color=%238957e5"> </a> <a href="/../../pulls?q=is%3Aopen+is%3Apr"> <img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr-raw/valmi-io/valmi-activation"> </a> <a href="/../../pulls?q=is%3Apr+is%3Aclosed"> <img alt="GitHub closed pull requests" src="https://img.shields.io/github/issues-pr-closed-raw/valmi-io/valmi-activation?color=%238957e5"> </a>
      --->
 </div>
<p align="center">valmi.io uses some of the best tools to create an Open Source reverse-ETL (data activation) Platform. It is built over the <a href="https://airbyte.com/">airbyte</a> protocol. <a href="https://www.getdbt.com/">dbt</a> is the centerpiece of our source connectors, and <a href="https://duckdb.org/">duckdb</a> for metrics. We engineered our orchestrator over <a href="https://dagster.io/">dagster</a>, and dagster dovetails perfectly with our vision of being a multi-persona tool.  </p>

  
- ### Checkout docs at - [https://www.valmi.io/docs/](https://www.valmi.io/docs/)
- ### Read the variety of usecases valmi.io can enable for your organization - [https://www.valmi.io/blog/](https://www.valmi.io/blog/)  
- ### 3 ways to start with valmi.io
    
    #### 1. Use it in the cloud
    - You can immediately create a sync at [https://demo.valmi.io.](https://demo.valmi.io/)
    - Watch a demo video to create a sync.
      
    <div align="center" >
  <br/>    

  [<img  src="https://blog.valmi.io/content/images/size/w1600/2023/06/Screenshot-2023-06-21-at-3.37.00-PM.png" width="70%" />](https://youtu.be/UBY0106gOD8 "Watch the demo video") 


    </div>
    
    #### 2.  Run it locally or in your Cloud
     - **Prerequisites:**
          valmi.io relies heavily on both Docker and Docker-compose. Install [Docker Desktop](https://docs.docker.com/compose/install/) to get both docker and docker-compose.
      
     - Clone this repo and recursively clone submodules.
       
       ```bash
       git clone git@github.com:valmi-io/valmi-activation.git
       cd valmi-activation
       git submodule update --init --recursive
       ```
    
    - Setup the environment.
      ```bash
      
      cp .env-example .env
      
      cd valmi-app-backend
      cp .env-example .env
      
      cd ../valmi-app
      `For macos`
      cp .env-example.macos .env
      `For linux`
      cp .env-example.linux .env
      ```
    
    - Intermediate storage, We are adding support for object stores like S3, GCS. Until then, Local storage is used.
      ```bash
      sudo mkdir -p /tmp/shared_dir/intermediate_store
      sudo chmod -R 777 /tmp/shared_dir/intermediate_store
      ```
    
    - Launch the reverse-etl service.
      ```bash
      ./valmi prod
      ```
      
    - To stop the service, run the following.
      ```bash
      ./valmi prod down
      ```
      
    - Please wait for about 2 minutes before you access the service, since valmi-app builds an optimized compiled version of the app UI. To access the service, please check the ['Accessing the service'](https://github.com/valmi-io/valmi-activation#accessing-the-service-for-local-deployments) section for local deployments.
    
    
    #### 3. Develop a connector locally to customize valmi.io as per your needs. You can just contact us too.
    
    - Clone, setup environment variables and create intermediate storage (see above section).
    - Create a new connector (Optional).
      ```bash
      # Copy code base from any existing connectors from valmi-integrations folder (ex. destination-webhook)
      
      cd valmi-integrations/connectors
      cp -r destination-webhook destination-awesome_connector
      
      # Make necessary changes and build the connector
      cd destination-awesome_connector
      make build_docker version=latest
      
      # Add the new connector information to "valmi-app-backend/init_db/connector_def.json"
      ```
    
    - Run the service.
      ```bash
      ./valmi dev
      ```
       
    - To access the service, please check the ['Accessing the service'](https://github.com/valmi-io/valmi-activation#accessing-the-service-for-local-deployments) section for local deployments.

    - To Stop the service, run the following.
      ```bash
      ./valmi dev down
      ```
    
- ### Accessing the service for local deployments
  
    Syncs <br> http://localhost:3000  |  Sync Runs <br> http://localhost:3000
    :-------------------------:|:-------------------------:
    ![](https://blog.valmi.io/content/images/size/w1000/2023/06/syncs_page.png)  |  ![](https://blog.valmi.io/content/images/size/w1000/2023/06/sync_runs_page.png)

    UI Backend Server API <br> http://localhost:4000/api/docs       |  Activation Server API <br> http://localhost:8000/docs
    :-------------------------:|:-------------------------:
    ![](https://blog.valmi.io/content/images/size/w1000/2023/06/app_backend_api.png)  |  ![](https://blog.valmi.io/content/images/size/w1000/2023/06/activation_server_api.png)

  
    

