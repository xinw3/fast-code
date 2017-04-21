package mapred.hashtagsim;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class HashtagCombiner extends Reducer<Text, Text, Text, Text> {

	@Override
	protected void reduce(Text key, Iterable<Text> value,
			Context context)
			throws IOException, InterruptedException {

		Map<String, Integer> tagMap = new HashMap<String, Integer>();
		for (Text word : value) {
			String w = word.toString();
			tagMap.put(w, tagMap.getOrDefault(w, 0) + 1);
		}

		/*
		 * We're serializing the word cooccurrence count as a string of the following form:
		 *
		 * word1:count1;word2:count2;...;wordN:countN;
		 */
		 StringBuilder sb = new StringBuilder();
		 for (String k : tagMap.keySet()) {
			 sb.append(k + ":" + tagMap.get(k) + ";");
		 }
		 context.write(key, new Text(sb.toString()));
	}
}
