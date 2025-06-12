import logging
from datetime import datetime

def setup_logger(name="MyAppLogger", log_file="debug.log", level=logging.DEBUG, mode='w'):
    # 如果已经存在该 logger，直接返回
    if name in logging.Logger.manager.loggerDict:
        return logging.getLogger(name)

    # 创建 logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = True  # 允许传播给父级（防止重复添加 handler）

    # 避免重复添加 handler
    if not logger.handlers:
        # 创建 Formatter
        formatter = logging.Formatter(
            fmt="===[%(name)s]=== - %(message)s",
            datefmt="%H:%M:%S"
        )

        # 创建 FileHandler
        file_handler = logging.FileHandler(log_file, mode=mode)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # 创建 StreamHandler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        # 添加 handler 到 logger
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger