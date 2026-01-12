package com.example.hello

import android.content.Context
import android.util.Log
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import org.pytorch.executorch.EValue
import java.io.File
import java.io.FileOutputStream

class ModelRunner(context: Context, private val tokenizer: TextTokenizer) {
    private var module: Module? = null
    var lastErrorMessage: String? = null

    init {
        try {
            val modelPath = assetFilePath(context , "deberta_classifier_weight_only.pte", force = true)
            module = Module.load(modelPath)
            Log.d("ModelRunner", "✅ Model loaded successfully from $modelPath")
        } catch (e: Exception) {
            lastErrorMessage = "Load Error: ${e.message}"
            Log.e("ModelRunner", "❌ Failed to load model", e)
        }
    }

    // New predict method that takes a String
    fun predict(inputText: String): FloatArray {
        val tokenized = tokenizer.tokenize(inputText)
        return predict(tokenized.tokenIds, tokenized.attentionMask)
    }

    // Existing predict method that takes LongArray inputs
    fun predict(tokenIds: LongArray, mask: LongArray): FloatArray {
        val currentModule = module
        if (currentModule == null) {
            Log.e("ModelRunner", "❌ Model is not loaded")
            return floatArrayOf()
        }

        try {
            val sequenceLength = tokenIds.size.toLong()
            val shape = longArrayOf(1, sequenceLength)

            val inputIdsTensor = Tensor.fromBlob(tokenIds, shape)
            val maskTensor = Tensor.fromBlob(mask, shape)

            val result = currentModule.forward(
                EValue.from(inputIdsTensor),
                EValue.from(maskTensor)
            )

            if (result.isEmpty()) {
                lastErrorMessage = "Inference Error: Model returned no outputs"
                return floatArrayOf()
            }

            val outputTensor = result[0].toTensor()
            return outputTensor.getDataAsFloatArray()
        } catch (e: Exception) {
            lastErrorMessage = "Inference Error: ${e.message}"
            Log.e("ModelRunner", "❌ Prediction failed", e)
            
            if (lastErrorMessage?.contains("arguments") == true) {
                Log.e("ModelRunner", "Hint: Your model might expect 3 inputs (token_type_ids?) instead of 2.")
            }
            return floatArrayOf()
        }
    }

    private fun assetFilePath(context: Context, assetName: String, force: Boolean = false): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && !force && file.length() > 0) return file.absolutePath

        context.assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                inputStream.copyTo(outputStream)
            }
        }
        return file.absolutePath
    }
}