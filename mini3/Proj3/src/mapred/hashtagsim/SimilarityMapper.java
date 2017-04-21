package mapred.hashtagsim;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class SimilarityMapper extends Mapper<LongWritable, Text, Text, IntWritable> {

	/**
	 * We compute the inner product of feature vector of every hashtag with that
	 * of #job
	 */
	@Override
	protected void map(LongWritable key, Text value, Context context)
			throws IOException, InterruptedException {
		String line = value.toString();
		String[] hashtag_featureVector = line.split("\\s+", 2);
		Map<String, Integer> tagMap = parseFeatureVector(hashtag_featureVector[1]);
		for (String k : tagMap.keySet()) {
				context.write(new Text(k), new IntWritable(tagMap.get(k)));
		}
	}

	/**
	 * De-serialize the feature vector into a map
	 *
	 * @param featureVector
	 *            The format is "word1:count1;word2:count2;...;wordN:countN;"
	 * @return A HashMap, ((#a, #b), 2)
	 */
	private Map<String, Integer> parseFeatureVector(String tagVector) {
		Map<String, Integer> tagMap = new HashMap<String, Integer>();
		String[] tags = tagVector.split(";");
		for (int i = 0; i < tags.length; i++) {
				String[] tag1 = tags[i].split(":");
				for (int j = i + 1; j < tags.length; j++) {
						StringBuilder sb = new StringBuilder();
						String[] tag2 = tags[j].split(":");
						sb.append(tag1[0]);
						sb.append(",");
						sb.append(tag2[0]);
						int count = Integer.parseInt(tag1[1]) * Integer.parseInt(tag2[1]);
						tagMap.put(sb.toString(), count);
				}
		}
		return tagMap;
	}
}
