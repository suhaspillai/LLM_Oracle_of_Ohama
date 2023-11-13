# Before running any modal command, its important to set up the environment before
It can be done using the following commands
- modal config show - This will show what is the value of environment variable - default its 'ENV'
```
(py310) suhaspillai@SuhasPillai:~/Suhas/git/llms/ask-fsdl$ modal config show
{'auto_snapshot': False,
 'automount': True,
 'default_cloud': None,
 'environment': 'ENV',  --- check this
 'function_runtime': None,
 'function_runtime_debug': False,
 'heartbeat_interval': 15,
 'image_id': None,
 'image_python_version': None,
 'loglevel': 'WARNING',
 'logs_timeout': 10,
 'outputs_timeout': 55.0,
 'profiling_enabled': False,
 'restore_state_path': '/opt/modal/restore-state.json',
 'serve_timeout': None,
 'server_url': 'https://api.modal.com',
 'sync_entrypoint': None,
 'task_id': None,
 'task_secret': None,
 'token_id': 'ak-oNi7UJkoWNO1fknfBNjdzZ',
 'token_secret': 'as-knpJtPcmi5PP9Bd21oEzQ4',
 'tracing_enabled': False,
 'worker_id': None}
```

# Set the environment varibale using the following command

```
modal config set-environment prod
```

# Run modal config show

```
(py310) suhaspillai@SuhasPillai:~/Suhas/git/llms/ask-fsdl$ modal config show{'auto_snapshot': False,
 'automount': True,
 'default_cloud': None,
 'environment': 'prod',
 'function_runtime': None,
 'function_runtime_debug': False,
 'heartbeat_interval': 15,
 'image_id': None,
 'image_python_version': None,
 'loglevel': 'WARNING',
 'logs_timeout': 10,
 'outputs_timeout': 55.0,
 'profiling_enabled': False,
 'restore_state_path': '/opt/modal/restore-state.json',
 'serve_timeout': None,
 'server_url': 'https://api.modal.com',
 'sync_entrypoint': None,
 'task_id': None,
 'task_secret': None,
 'token_id': 'ak-oNi7UJkoWNO1fknfBNjdzZ',
 'token_secret': 'as-knpJtPcmi5PP9Bd21oEzQ4',
 'tracing_enabled': False,
 'worker_id': None}

 ```

# For error NotFoundError: No secret named mongodb-fsdl - you can add secrets to your 

- Run make secrets

# To check what is stored in mongodb mongoDB
- In mongo DB collections are nothing but Tables

-Starting mongoDB instance
- mongodb+srv://<credentials>@fsdl.1jw1q9u.mongodb.net/?appName=mongosh+1.10.6

- show dbs
- use database
- db.collections() -  Gives the collections in the dataase
- db['<collectionname>'].find() - This will list all that is tored in the collections

