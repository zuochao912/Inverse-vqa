"""Transform all the IQ VQA dataset into a hdf5 dataset.
"""

from PIL import Image
from torchvision import transforms

import argparse
import json
import h5py
import numpy as np
import os
import progressbar

from train_utils import Vocabulary
from vocab import load_vocab
from vocab import process_text

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_answer_mapping(annotations, ans2cat):
    """Returns mapping from question_id to answer.

    Only returns those mappings that map to one of the answers in ans2cat.

    Args:
        annotations: VQA annotations file.
        ans2cat: Map from answers to answer categories that we care about.

    Returns:
        answers: Mapping from question ids to answers.
        image_ids: Set of image ids.
    """
    answers = {}
    image_ids = set()
    for q in annotations['annotations']:
        question_id = q['question_id']
        answer = q['multiple_choice_answer']
        if answer in ans2cat:
            answers[question_id] = answer
            image_ids.add(q['image_id'])
    return answers, image_ids


def save_dataset(image_dir, questions_path, annotations, vocab, ans2cat, output,
                 im_size=224, max_q_length=20, max_a_length=4,
                 with_answers=False):
    """Saves the Visual Genome images and the questions in a hdf5 file.

    Args:
        image_dir: Directory with all the images.
        questions: Location of the questions.
        annotations: Location of all the annotations.
        vocab: Location of the vocab file.
        ans2cat: Mapping from answers to category.
        output: Location of the hdf5 file to save to.
        im_size: Size of image.
        max_q_length: Maximum length of the questions.
        max_a_length: Maximum length of the answers.
        with_answers: Whether to also save the answers.
    """
    # Load the data.
    vocab = load_vocab(vocab)
    with open(annotations) as f:
        annos = json.load(f)
    with open(questions_path) as f:
        questions = json.load(f)

    # Get the mappings from qid to answers.
    qid2ans, image_ids = create_answer_mapping(annos, ans2cat)
    total_questions = len(qid2ans.keys())
    total_images = len(image_ids)
    print ("Number of images to be written: %d" % total_images)
    print ("Number of QAs to be written: %d" % total_questions)

    h5file = h5py.File(output, "w")
    d_questions = h5file.create_dataset(
        "questions", (total_questions, max_q_length), dtype='i')
    d_indices = h5file.create_dataset(
        "image_indices", (total_questions,), dtype='i')
    d_images = h5file.create_dataset(
        "images", (total_images, im_size, im_size, 3), dtype='f')
    d_answers = h5file.create_dataset(
        "answers", (total_questions, max_a_length), dtype='i')
    d_answer_types = h5file.create_dataset(
        "answer_types", (total_questions,), dtype='i')

    # Create the transforms we want to apply to every image.
    transform = transforms.Compose([
        transforms.Resize((im_size, im_size))])

    # Iterate and save all the questions and images.
    bar = progressbar.ProgressBar(maxval=total_questions)
    i_index = 0
    q_index = 0
    done_img2idx = {}
    notfoundlist=set()
    #将训练集、验证集中的所有图片，读取，变换后，直接放到d_images这个h5文件之中
    for entry in questions['questions']:
        image_id = entry['image_id']
        question_id = entry['question_id']
        if image_id not in image_ids:
            continue
        if question_id not in qid2ans:
            continue
        if image_id not in done_img2idx:
            try:
                path = "COCO_val2014_%d.jpg" % (image_id)
                #COCO_train2014_
                image = Image.open(os.path.join(image_dir, path)).convert('RGB')
            except IOError:
                path = "COCO_val2014_%012d.jpg" % (image_id)
                #COCO_train2014_
                image = Image.open(os.path.join(image_dir, path)).convert('RGB')

            image = transform(image)
            d_images[i_index, :, :, :] = np.array(image) #直接存取
            done_img2idx[image_id] = i_index  #img_id到h5文件的下标
            i_index += 1
        q, length = process_text(entry['question'], vocab,
                                 max_length=max_q_length) #问题最长默认20，经过tokenize，有首尾标记;返回的是numpy数组
        d_questions[q_index, :length] = q
        answer = qid2ans[question_id]
        a, length = process_text(answer, vocab,
                                 max_length=max_a_length) #有首尾标记，答案默认最长是4；返回的是numpy数组；值得注意的是，答案居然不是分类问题？
        d_answers[q_index, :length] = a  #h5问题下标：问题，问题是tokenize后的numpy矩阵
        d_answer_types[q_index] = int(ans2cat[answer])#h5问题下标：问题类型下标
        d_indices[q_index] = done_img2idx[image_id] #h5问题下标：h5图片下标
        q_index += 1
        bar.update(q_index)
    h5file.close()
    print ("Number of images written: %d" % i_index)
    print ("Number of QAs written: %d" % q_index)
    print("We have {} images not found,they are listed below".format(len(notfoundlist)))
    print(notfoundlist)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Inputs.
    parser.add_argument('--image-dir', type=str, default='../data/val2014',
                        help='directory for resized images')
                        #../data/train2014
    parser.add_argument('--questions', type=str,
                        default='../data/v2_OpenEnded_mscoco_val2014_questions.json',
                        help='Path for train annotation file.')
                        #v2_OpenEnded_mscoco_train2014_questions.json
    parser.add_argument('--annotations', type=str,
                        default='../data/v2_mscoco_'
                        'val2014_annotations.json',
                        help='Path for train annotation file.')
                        #v2_mscoco_train2014_annotations.json
    parser.add_argument('--cat2ans', type=str,
                        default='./data/iq_dataset.json',
                        help='Path for the answer types.')
    parser.add_argument('--vocab-path', type=str,
                        default='data/processed/vocab_iq.json',
                        help='Path for saving vocabulary wrapper.')

    # Outputs.
    parser.add_argument('--output', type=str,
                        default='data/processed/iq_val_dataset.hdf5',
                        help='directory for resized images.')
                        #iq_dataset.hdf5
    parser.add_argument('--cat2name', type=str,
                        default='data/processed/cat2name.json',
                        help='Location of mapping from category to type name.')

    # Hyperparameters.
    parser.add_argument('--im_size', type=int, default=224,
                        help='Size of images.')
    parser.add_argument('--max-q-length', type=int, default=20,
                        help='maximum sequence length for questions.')
    parser.add_argument('--max-a-length', type=int, default=4,
                        help='maximum sequence length for answers.')
    args = parser.parse_args()

    ans2cat = {}
    with open(args.cat2ans) as f:
        cat2ans = json.load(f)
    cats = sorted(cat2ans.keys())
    #读取了类别-答案的映射词典，并将类别提取，排序；后续只选择这些类别答案的进行生成。
    with open(args.cat2name, 'w') as f:
        json.dump(cats, f)
    for cat in cat2ans:
        for ans in cat2ans[cat]:
            ans2cat[ans] = cats.index(cat)
            #以上对键cat进行遍历，对值ans遍历，ans2cat是获得答案类别在cats排序后的列表的下标
    save_dataset(args.image_dir, args.questions, args.annotations, args.vocab_path,
                 ans2cat, args.output, im_size=args.im_size,
                 max_q_length=args.max_q_length, max_a_length=args.max_a_length)
    print('Wrote dataset to %s' % args.output)
    # Hack to avoid import errors.
    Vocabulary()
