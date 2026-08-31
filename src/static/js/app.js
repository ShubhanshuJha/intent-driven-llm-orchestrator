const queryInput = document.getElementById("query-input");
const sendButton = document.getElementById("send-button");
const chatContainer = document.getElementById("chat-container");
const loading = document.getElementById("loading");


queryInput.addEventListener("keydown", function (event) {

    if (
        event.key === "Enter" &&
        !event.shiftKey
    ) {
        event.preventDefault();

        sendQuery();
    }
});


async function sendQuery() {

    const query = queryInput.value.trim();

    if (!query) {
        return;
    }


    addMessage(
        "user",
        query
    );


    queryInput.value = "";

    setLoading(true);


    try {

        const response = await fetch(
            "/api/v1/query",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    query: query
                })
            }
        );


        if (!response.ok) {

            const error = await response.json();

            throw new Error(
                error.detail || "Request failed"
            );
        }


        const data = await response.json();


        addMessage(
            "assistant",
            data.response
        );

    }
    catch (error) {

        addMessage(
            "assistant",
            `Error: ${error.message}`
        );

    }
    finally {

        setLoading(false);

    }
}


// function addMessage(
//     role,
//     content,
//     metadata = null
// ) {

//     const message = document.createElement("div");

//     message.className =
//         `message ${role}-message`;


//     const contentElement =
//         document.createElement("div");

//     contentElement.className =
//         "message-content";

//     contentElement.textContent =
//         content;


//     if (
//         role === "assistant" &&
//         metadata
//     ) {

//         const metaElement =
//             document.createElement("div");

//         metaElement.className =
//             "metadata";

//         metaElement.innerHTML = `
//             Model: ${metadata.selected_model}
//             &nbsp; | &nbsp;
//             Attempts: ${metadata.attempts}
//             &nbsp; | &nbsp;
//             Judge: ${metadata.evaluation.verdict}
//             &nbsp; | &nbsp;
//             Score: ${metadata.evaluation.score}/10
//         `;


//         contentElement.appendChild(
//             metaElement
//         );
//     }


//     message.appendChild(
//         contentElement
//     );

//     chatContainer.appendChild(
//         message
//     );


//     chatContainer.scrollTop =
//         chatContainer.scrollHeight;
// }
function addMessage(
    role,
    content
) {
    const message = document.createElement("div");

    message.className =
        `message ${role}-message`;

    const contentElement =
        document.createElement("div");

    contentElement.className =
        "message-content";


    if (role === "assistant") {
        contentElement.innerHTML =
            marked.parse(content);
    } else {
        contentElement.textContent =
            content;
    }


    message.appendChild(
        contentElement
    );

    chatContainer.appendChild(
        message
    );


    if (
        role === "assistant" &&
        typeof renderMathInElement !== "undefined"
    ) {
        renderMathInElement(
            contentElement,
            {
                delimiters: [
                    {
                        left: "$$",
                        right: "$$",
                        display: true
                    },
                    {
                        left: "$",
                        right: "$",
                        display: false
                    },
                    {
                        left: "\\(",
                        right: "\\)",
                        display: false
                    },
                    {
                        left: "\\[",
                        right: "\\]",
                        display: true
                    }
                ],

                throwOnError: false
            }
        );
    }


    chatContainer.scrollTop =
        chatContainer.scrollHeight;
}


function setLoading(state) {

    sendButton.disabled = state;

    loading.classList.toggle(
        "hidden",
        !state
    );
}
