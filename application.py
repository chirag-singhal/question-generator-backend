from flask import Flask, request, jsonify
from generate_questions import generate_question, get_entities
from flask_cors import cross_origin

app = Flask(__name__)

@app.route('/generate_questions', methods=['POST'])
@cross_origin()
def generate_questions():
    try:
        paragraph = request.json['paragraph']
        entities = get_entities(paragraph)
        if not entities:
            return jsonify({'error': 'No entities found in the given paragraph.'}), 400

        questions = []
        for entity in entities:
            print(entities)
            question = generate_question(paragraph, entity.text)
            questions.append(question)
        
        return jsonify({'questions': questions})
    except KeyError:
        return jsonify({'error': 'No paragraph provided in the request.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)