import src.tasks.predict_generalizability_from_saved_models as generalizability
import src.tasks.main as main_task

def main():
    main_task.main()
    generalizability.main()

if __name__ == "__main__":
    main()
