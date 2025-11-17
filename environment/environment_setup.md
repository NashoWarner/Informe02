# Configuración del entorno

1. Instale anaconda o miniconda.
2. Cree un entorno conda a partir de los archivos .yml que se encuentran en la carpeta `/environment`:
    - Si está ejecutando Windows, use el símbolo del sistema de Conda; en Mac o Linux, puede usar la Terminal.
    - Use el comando: `conda env create -f inf02_env_<OS>.yml`
    - Asegúrese de modificar el comando según su sistema operativo (`linux`, `mac` o `win`).
    - Esto debería crear un entorno llamado `inf02`.
3. Active el entorno conda:
    - Comando de Windows: `activate inf02`
    - Comando de MacOS/Linux: `conda activate inf02`

For more references on conda environments, refer to [Conda Managing Environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or the [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)