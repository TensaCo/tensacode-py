"""Load an owned TensorCode chatbot from a local artifact or Hugging Face Hub."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', help='Local model directory or Hugging Face repository ID')
    parser.add_argument('--revision')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--prompt', help='One conversation turn; omit for an interactive session')
    parser.add_argument('--load-session')
    parser.add_argument('--save-session')
    args = parser.parse_args()
    from tensorcode.tools.chatbot import Chatbot
    model = Chatbot.from_pretrained(args.model, revision=args.revision, device=args.device,
                                   local_files_only=args.local_files_only)
    if args.load_session:
        model.load_session(args.load_session)
    if args.prompt is not None:
        print(model(args.prompt))
    else:
        while True:
            try:
                prompt = input('You: ')
            except (EOFError, KeyboardInterrupt):
                break
            if prompt.strip():
                print('Chatbot:', model(prompt))
    if args.save_session:
        model.save_session(args.save_session)


if __name__ == '__main__':
    main()
