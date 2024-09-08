---
title: "Convert Latex file to markdown including images"
themes: ["projects"]
tags: ["Document Conversion", "Markdown", "Latex"]
---

<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>arXiv Paper Viewer</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/axios/0.21.1/axios.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/2.0.3/marked.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        input, button {
            font-size: 16px;
            padding: 10px;
            margin: 10px 0;
        }
        #output {
            border: 1px solid #ddd;
            padding: 20px;
            margin-top: 20px;
            white-space: pre-wrap;
        }
        .highlight-button {
            border: 1px solid black;
            background-color: red;
            color: black;
            width: auto;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s, color 0.3s;
            margin-top: 30px;
        }
        .highlight-button:hover {
            background-color: #ff6666; /* Rouge plus clair pour la surbrillance */
            color: black;
        }
    </style>
</head>
<body>
    <h1>arXiv Paper Viewer</h1>
    <input type="text" id="arxivId" placeholder="Enter arXiv paper ID">
    <button onclick="submitQuestion()" class="highlight-button">Run</button>
    <div id="output"></div>

    <script>
        async function runProcess() {
            const arxivId = document.getElementById('arxivId').value;
            const outputDiv = document.getElementById('output');
            
            outputDiv.innerHTML = 'Processing...';
            
            try {
                const response = await axios.post('/process', { arxivId: arxivId });
                const markdown = response.data;
                
                // Convert Markdown to HTML and display
                outputDiv.innerHTML = marked(markdown);
            } catch (error) {
                outputDiv.innerHTML = `Error: ${error.message}`;
            }
        }
    </script>
</body>
</html>