import logging

import sys
def setup_logging(script_name):
    try:
        logger = logging.getLogger(script_name)
        if not logger.handlers:
            logger.setLevel(logging.DEBUG)
            # Create a file handler for the script
            handler = logging.FileHandler(f'C:\\Users\\sabda\\Downloads\\Customer Retention Prediction System\\logs\\{script_name}.log', mode='w')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = False

        return logger

    except Exception as e:  #EXCEPTIONAL(ERROR HANDLING METHOD)
        err_type,err_line,err_msg = sys.exc_info()
        logger.info(f"ERROR TYPE: {err_type} : CAUSE : {err_msg} : IN LINE: {err_line.tb_lineno}")