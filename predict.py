import argparse
from bert.predire import predire_binaire


def argument_parser():
    parser = argparse.ArgumentParser(usage='\n python predict.py [model] [model_dir] [data_path] [dataset]'
                                           '\n python predict.py --model=true --model_dir=./modeles/bert_true --data_path=./donnees/FakeNews_Task3_2022_V0/English_data_test_release_with_rating.csv --dataset=test',
                                     description="Ce script permet d'obtenir des prédictions des différents modèles de classification binaire.",
                                     add_help=True)
    parser.add_argument('--model', type=str, default="None", choices=["None", "false", "other", "partfalse", "true"],
                        help="Tâche de classification")
    parser.add_argument('--model_dir', type=str, default="None",
                        help="Chemin du modèle.")
    parser.add_argument('--data_path', type=str, default="None",
                        help="Chemin des données.")
    parser.add_argument('--dataset', type=str, default="None", choices=["None", "train", "dev", "test"],
                        help="Ensemble de données.")
    parser.add_argument('--results_dir', type=str, default="./donnees/resultats/",
                        help="Chemin de sauvegarde des résultats.")
    return parser.parse_args()


if __name__ == "__main__":
    
    args = argument_parser()

    if args.model=="None":
        print("Aucun modèle spécifié")
        exit(0)
    if args.model_dir=="None":
        print("Aucun chemin de modèle spécifié")
        exit(0)
    if args.data_path=="None":
        print("Aucune données spécifiées")
        exit(0)
    if args.dataset=="None":
        print("Aucun ensemble de données spécifié")
        exit(0)
    
    print(f"Prédictions du modèle {args.model}")
    label_map = {"false": 0, "other": 1, "partfalse": 2, "true": 3}

    predire_binaire(args.data_path,args.results_dir,args.model_dir,label_map[args.model],args.dataset)
