package com.example.hello

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.facebook.soloader.SoLoader
import android.util.Log

class MainActivity : AppCompatActivity() {

    private lateinit var inputEditText: EditText
    private lateinit var predictButton: Button
    private lateinit var resultTextView: TextView
    private lateinit var modelRunner: ModelRunner

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        try {
            SoLoader.init(this, false)
        } catch (e: Exception) {
            Log.e("MainActivity", "SoLoader initialization failed", e)
        }

        setContentView(R.layout.activity_main)
        
        inputEditText = findViewById(R.id.inputEditText)
        predictButton = findViewById(R.id.predictButton)
        resultTextView = findViewById(R.id.resultTextView)
        resultTextView.text = "Enter text and press Predict."

        try {
            // 1. Initialize Real Tokenizer
            val tokenizer = DebertaTokenizer(this, "vocab.txt")
            
            // 2. Initialize Model Runner
            modelRunner = ModelRunner(this, tokenizer)
            
            predictButton.setOnClickListener {
                val inputText = inputEditText.text.toString()
                if (inputText.isBlank()) {
                    resultTextView.text = "Please enter some text."
                    return@setOnClickListener
                }
                
                resultTextView.text = "Processing..."
                
                try {
                    val logits = modelRunner.predict(inputText)

                    if (logits.isNotEmpty()) {
                        // Assuming 0 is Negative, 1 is Positive
                        val negScore = logits.getOrElse(0) { 0f }
                        val posScore = logits.getOrElse(1) { 0f }
                        
                        val sentiment = if (posScore > negScore) "Positive 😊" else "Negative ☹️"
                        val result = "Sentiment: $sentiment\n\nRaw Logits:\nNeg: $negScore\nPos: $posScore"
                        
                        resultTextView.text = result
                    } else {
                        val error = modelRunner.lastErrorMessage ?: "Unknown error"
                        resultTextView.text = "Prediction failed:\n$error"
                    }
                } catch (e: Exception) {
                    resultTextView.text = "Error: ${e.message}"
                    Log.e("MainActivity", "Inference crash", e)
                }
            }
        } catch (e: Exception) {
            resultTextView.text = "Initialization Error: ${e.message}\n(Make sure vocab.txt is in assets)"
            Log.e("MainActivity", "Init failed", e)
        }
    }
}