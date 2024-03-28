import { useState } from "react"

const options = {
    method: 'POST',
    headers: {
        'content-type': 'application/json'
    },
    body: JSON.stringify({
        "acrostic": "hello",
        "delimiter": "-",
        "min_length": 8,
        "max_length": 12
    })
}

export default function PasswordGenerator() {
    const [password, setPassword] = useState('')

    function handleGenerate() {
        console.log('handleGenerate')
        fetch('http://localhost:5000/generate', options)
            .then(res => res.json())
            .then(res => {
                setPassword(res.password)
            })
    }

    return (
        <div className="container d-flex">
        <div className="container py-4">
            <div className="row mb-2">
                <div className="col-12">
                    <h1 className="display-4">Password Generator</h1>
                </div>
        {password && 
            <input type="text" class="form-control mt-2" value={password} readonly style={{display: password ? 'inline-block' : 'none'}} />
        }
        &nbsp;

        <button class="btn btn-primary mt-3" onClick={handleGenerate}>Generate</button>
        
        </div>
        </div>


        </div>
    )
}