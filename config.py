from dynaconf import Dynaconf

configs = Dynaconf(
    envvar_prefix="",
    settings_files=['configs/configs.toml', '.secrets.toml'],
)
