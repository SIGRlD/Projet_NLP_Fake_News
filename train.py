import argparse
from bert.entrainer import entrainer_binaire


def argument_parser():
    parser = argparse.ArgumentParser(usage='\n python train.py [model] [parameters]'
                                           '\n python train.py --model=true'
                                           '\n python train.py --model=true --model_dir=./modeles/bert_true'
                                           '\n python train.py --model=true --model_dir=./modeles/bert_true --add_data',
                                     description="Ce script permet d'entrainer les différents modèles de classification binaire.",
                                     add_help=True)
    parser.add_argument('--model', type=str, default="None", choices=["None", "false", "other", "partfalse", "true"],
                        help="Tâche de classification")
    parser.add_argument('--train_data', type=str, default="./donnees/FakeNews_Task3_2022_V0/Task3_english_training.csv",
                        help="Chemin des données d'entrainement.")
    parser.add_argument('--dev_data', type=str, default="./donnees/FakeNews_Task3_2022_V0/Task3_english_dev.csv",
                        help="Chemin des données de validation.")
    parser.add_argument('--model_dir', type=str, default="./modeles/bert",
                        help="Chemin de sauvegarde du modèle.")
    parser.add_argument('--add_data', action='store_true',
                        help="Si on ajoute des données pour l'entrainement")
    return parser.parse_args()


if __name__ == "__main__":
    
    args = argument_parser()

    if args.model=="None":
        print("Aucun modèle spécifié")
        exit(0)
    
    print(f"Entrainement du modèle {args.model}")
    label_map = {"false": 0, "other": 1, "partfalse": 2, "true": 3}

    entrainer_binaire(args.train_data,args.dev_data,args.model_dir,label_map[args.model],args.add_data)
