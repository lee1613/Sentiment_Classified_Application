package com.example.hello

import android.content.Context
import java.io.BufferedReader
import java.io.InputStreamReader

interface TextTokenizer {
    fun tokenize(text: String): TokenizationResult
}

data class TokenizationResult(
    val tokenIds: LongArray,
    val attentionMask: LongArray
)

class DebertaTokenizer(context: Context, vocabAsset: String) : TextTokenizer {
    private val vocab: Map<String, Int>
    private val CLS_TOKEN = "[CLS]"
    private val SEP_TOKEN = "[SEP]"
    private val UNK_TOKEN = "[UNK]"
    
    // With dynamic shapes, we can now support longer sequences!
    // 128 is a common standard, but you can go higher if your model supports it.
    private val MAX_SEQ_LEN = 128 

    init {
        val tempVocab = mutableMapOf<String, Int>()
        context.assets.open(vocabAsset).use { inputStream ->
            BufferedReader(InputStreamReader(inputStream)).useLines { lines ->
                lines.forEachIndexed { index, token ->
                    tempVocab[token] = index
                }
            }
        }
        vocab = tempVocab
    }

    override fun tokenize(text: String): TokenizationResult {
        val tokens = mutableListOf<String>()
        tokens.add(CLS_TOKEN)

        // Simple WordPiece tokenization logic
        val words = text.lowercase().split(Regex("\\s+"))
        for (word in words) {
            tokens.addAll(wordPieceTokenize(word))
        }

        // Handle truncation and SEP addition
        val finalTokens = if (tokens.size >= MAX_SEQ_LEN) {
            val truncated = tokens.take(MAX_SEQ_LEN - 1).toMutableList()
            truncated.add(SEP_TOKEN)
            truncated
        } else {
            tokens.add(SEP_TOKEN)
            tokens
        }

        return convertToResult(finalTokens)
    }

    private fun convertToResult(finalTokens: List<String>): TokenizationResult {
        // Even with dynamic shapes, it's safer to pad to a consistent size 
        // to avoid constant memory reallocations in the engine.
        val tokenIds = LongArray(MAX_SEQ_LEN) { 0L }
        val attentionMask = LongArray(MAX_SEQ_LEN) { 0L }

        for (i in 0 until MAX_SEQ_LEN) {
            if (i < finalTokens.size) {
                tokenIds[i] = vocab[finalTokens[i]]?.toLong() ?: vocab[UNK_TOKEN]?.toLong() ?: 0L
                attentionMask[i] = 1L
            } else {
                // Padding (DeBERTa typically uses 0 for padding)
                tokenIds[i] = 0L 
                attentionMask[i] = 0L
            }
        }
        return TokenizationResult(tokenIds, attentionMask)
    }

    private fun wordPieceTokenize(word: String): List<String> {
        val outputTokens = mutableListOf<String>()
        var start = 0
        while (start < word.length) {
            var end = word.length
            var curSubword = ""
            while (start < end) {
                var substr = word.substring(start, end)
                if (start > 0) substr = "##$substr"
                if (vocab.containsKey(substr)) {
                    curSubword = substr
                    break
                }
                end--
            }
            if (curSubword.isEmpty()) {
                outputTokens.add(UNK_TOKEN)
                break
            }
            outputTokens.add(curSubword)
            start = end
        }
        return outputTokens
    }
}